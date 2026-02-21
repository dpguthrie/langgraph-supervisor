# Azure App Service Deployment for Braintrust Eval Server

This guide covers deploying the Braintrust remote eval server as a Docker container on **Azure App Service for Containers**. This is an alternative to the [Modal deployment](../src/eval_server.py) that gives you a standard Docker image you can run anywhere.

## How It Works

The deployment consists of three pieces:

| File | Role |
|------|------|
| `azure_app.py` | ASGI application entry point. Loads all `eval_*.py` evaluators from the `evals/` directory and exposes a Starlette app via Braintrust's `create_app()`. |
| `Dockerfile` | Builds a `python:3.13-slim` image with all dependencies installed via `uv`. |
| `docker-entrypoint.sh` | Starts gunicorn with uvicorn workers, binding to Azure's `$PORT` (defaults to 8000). |

### Startup sequence

1. **Parameter patch** &mdash; `azure_app.py` imports `evals.braintrust_parameter_patch` and calls `apply_parameter_patch()` before any Braintrust code runs. This ensures the SDK correctly handles Pydantic parameter models in evaluators.
2. **Evaluator discovery** &mdash; All `evals/eval_*.py` files are globbed and loaded via Braintrust's `EvaluatorState` / `update_evaluators`.
3. **ASGI app** &mdash; `create_app(evaluators)` produces a Starlette application that serves the evaluator list, accepts eval run requests, etc.
4. **Gunicorn** &mdash; The entrypoint starts gunicorn pointing at `azure_app:app` with uvicorn async workers.

## Prerequisites

- Docker
- An Azure account (for cloud deployment)
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) (`az`) installed
- API keys: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `BRAINTRUST_API_KEY`

## Local Development

### Build and run locally

```bash
# Build the image
docker build -t langgraph-eval-server .

# Run with your environment variables
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e TAVILY_API_KEY="$TAVILY_API_KEY" \
  -e BRAINTRUST_API_KEY="$BRAINTRUST_API_KEY" \
  langgraph-eval-server
```

### Verify it's working

```bash
# Health check
curl http://localhost:8000/

# List loaded evaluators
curl http://localhost:8000/list
```

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `PORT` | `8000` | Port the server listens on (Azure sets this automatically) |
| `GUNICORN_WORKERS` | `2` | Number of gunicorn worker processes |
| `OPENAI_API_KEY` | &mdash; | Required for LLM calls |
| `TAVILY_API_KEY` | &mdash; | Required for research agent web search |
| `BRAINTRUST_API_KEY` | &mdash; | Required for Braintrust logging and eval |

## Deploy to Azure App Service

### 1. Create an Azure Container Registry (one-time)

```bash
RESOURCE_GROUP="my-resource-group"
ACR_NAME="myregistry"           # must be globally unique
LOCATION="eastus"

az group create --name $RESOURCE_GROUP --location $LOCATION
az acr create --name $ACR_NAME --resource-group $RESOURCE_GROUP --sku Basic --admin-enabled true
```

### 2. Build and push the image

```bash
# Option A: Build in Azure (no local Docker needed)
az acr build --registry $ACR_NAME --image langgraph-eval-server:latest .

# Option B: Build locally and push
az acr login --name $ACR_NAME
docker build -t $ACR_NAME.azurecr.io/langgraph-eval-server:latest .
docker push $ACR_NAME.azurecr.io/langgraph-eval-server:latest
```

### 3. Create the App Service

```bash
APP_NAME="langgraph-eval-server"    # must be globally unique
APP_PLAN="eval-server-plan"

# Create an App Service Plan (B1 or higher recommended)
az appservice plan create \
  --name $APP_PLAN \
  --resource-group $RESOURCE_GROUP \
  --is-linux \
  --sku B1

# Create the web app with the container image
az webapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --plan $APP_PLAN \
  --deployment-container-image-name "$ACR_NAME.azurecr.io/langgraph-eval-server:latest"
```

### 4. Configure environment variables

```bash
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    OPENAI_API_KEY="your-key" \
    TAVILY_API_KEY="your-key" \
    BRAINTRUST_API_KEY="your-key" \
    WEBSITES_PORT=8000
```

### 5. Configure timeouts and availability

Eval runs can take several minutes, so the default Azure timeout (230s) is too short.

```bash
# Set request timeout to 3600 seconds
az webapp config set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --generic-configurations '{"requestTimeout": "3600"}'

# Enable Always On to prevent container recycling (requires Basic tier or higher)
az webapp config set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --always-on true
```

### 6. Verify the deployment

```bash
APP_URL="https://$APP_NAME.azurewebsites.net"

# Health check
curl "$APP_URL/"

# List evaluators
curl "$APP_URL/list"
```

## Connect from Braintrust Playground

1. Open the [Braintrust Playground](https://www.braintrust.dev/app)
2. Go to your project's eval configuration
3. Set the remote eval server URL to your App Service URL:
   ```
   https://<APP_NAME>.azurewebsites.net
   ```
4. Run an eval &mdash; it will execute on your Azure container

## Updating the Deployment

After making code changes:

```bash
# Rebuild and push
az acr build --registry $ACR_NAME --image langgraph-eval-server:latest .

# Restart the app to pick up the new image
az webapp restart --name $APP_NAME --resource-group $RESOURCE_GROUP
```

## Troubleshooting

### View container logs

```bash
az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP
```

### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Container fails to start | Missing env vars | Check Application Settings in Azure Portal |
| 502 Bad Gateway after deploy | App still starting (cold start) | Wait 30-60s; enable Always On |
| Eval times out | Default 230s Azure timeout | Set `requestTimeout` to 3600 (see step 5) |
| No evaluators listed at `/list` | `evals/` directory missing eval files | Verify `evals/eval_*.py` files are in the Docker image |

## Architecture Comparison

| | Modal (`src/eval_server.py`) | Azure (`azure_app.py` + Docker) |
|---|---|---|
| **Runtime** | Modal serverless | Azure App Service container |
| **Scaling** | Auto (Modal manages) | Manual (App Service Plan tier) |
| **Cold starts** | `min_containers=1` keeps warm | Always On setting keeps warm |
| **Secrets** | `modal.Secret.from_dotenv()` | Azure Application Settings |
| **Timeout** | 3600s via `@app.function(timeout=)` | 3600s via `requestTimeout` config |
| **Portability** | Modal-specific | Any Docker host |

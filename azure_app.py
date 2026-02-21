"""Azure App Service entry point for Braintrust remote eval dev server.

This is the Docker/Azure equivalent of src/eval_server.py (Modal).
Gunicorn loads this module as `azure_app:app`.

Deployment steps:
  1. docker build -t langgraph-eval-server .
  2. Push to Azure Container Registry (az acr build or docker push)
  3. Create App Service: az webapp create --deployment-container-image-name ...
  4. Set Application Settings: OPENAI_API_KEY, TAVILY_API_KEY,
     BRAINTRUST_API_KEY, WEBSITES_PORT=8000
  5. Enable Always On and set request timeout to 3600s
  6. Connect from Braintrust Playground using the App Service URL
"""

from pathlib import Path

# IMPORTANT: Apply the SDK patch BEFORE any Braintrust imports
# This ensures the patched version is used when evaluators are loaded
from evals.braintrust_parameter_patch import apply_parameter_patch

apply_parameter_patch()

# Now import Braintrust components (they will use the patched version)
from braintrust.cli.eval import EvaluatorState, FileHandle, update_evaluators
from braintrust.devserver.server import create_app

# Find all eval files in the evals directory
evals_dir = Path(__file__).resolve().parent / "evals"
eval_files = sorted(evals_dir.glob("eval_*.py"))
print(f"Found {len(eval_files)} eval file(s): {[f.name for f in eval_files]}")

# Load evaluators using Braintrust's CLI loader
handles = [FileHandle(in_file=str(eval_file)) for eval_file in eval_files]
eval_state = EvaluatorState()
update_evaluators(eval_state, handles, terminate_on_failure=True)
evaluators = [e.evaluator for e in eval_state.evaluators]

print(f"Loaded {len(evaluators)} evaluator(s): {[e.eval_name for e in evaluators]}")

# Create the ASGI app — gunicorn picks this up as azure_app:app
app = create_app(evaluators, org_name=None)

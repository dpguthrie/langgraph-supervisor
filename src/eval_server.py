"""Modal deployment for Braintrust remote eval dev server.

This is a simplified version that uses Braintrust's built-in create_app()
function instead of manually recreating all the middleware and routing logic.
"""

import modal

# Create image with all dependencies
modal_image = (
    modal.Image.debian_slim()
    .apt_install(
        "git"
    )  # Git is required for installing braintrust from git branch (temporary remote evals fix)
    .uv_sync()  # Install dependencies from pyproject.toml/requirements.txt
    .add_local_python_source("src")
    .add_local_python_source("evals")
)

app = modal.App("langgraph-supervisor-eval-server", image=modal_image)

# Always read secrets from local .env and send them as a Secret
_secrets = [modal.Secret.from_dotenv()]


@app.function(
    secrets=_secrets,
    # Keep the server warm with at least 1 instance
    min_containers=1,
    # Timeout for long-running evals
    timeout=3600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def braintrust_eval_server():
    """
    Run Braintrust remote eval dev server on Modal.

    This uses Braintrust's built-in create_app() function which handles
    all the routing, middleware, and ASGI app setup automatically.
    """
    from pathlib import Path

    # IMPORTANT: Apply the SDK patch BEFORE any Braintrust imports
    # This ensures the patched version is used when evaluators are loaded
    from evals.braintrust_parameter_patch import apply_parameter_patch

    apply_parameter_patch()

    # Now import Braintrust components (they will use the patched version)
    from braintrust.cli.eval import EvaluatorState, FileHandle, update_evaluators
    from braintrust.devserver.server import create_app

    import evals

    # Find all eval files in the evals directory
    # In Modal, the evals package is mounted and importable
    if hasattr(evals, "__path__") and evals.__path__:
        evals_dir = Path(evals.__path__[0])
    elif hasattr(evals, "__file__") and evals.__file__:
        evals_dir = Path(evals.__file__).parent
    else:
        raise RuntimeError("Could not locate evals package directory")

    print(f"Scanning for evaluators in {evals_dir}")

    # Find all eval_*.py files (matching braintrust CLI pattern)
    eval_files = list(evals_dir.glob("eval_*.py"))
    print(f"Found {len(eval_files)} eval file(s): {[f.name for f in eval_files]}")

    # Load evaluators using Braintrust's CLI loader
    handles = [FileHandle(in_file=str(eval_file)) for eval_file in eval_files]
    eval_state = EvaluatorState()
    update_evaluators(eval_state, handles, terminate_on_failure=True)
    evaluators = [e.evaluator for e in eval_state.evaluators]

    print(f"Loaded {len(evaluators)} evaluator(s): {[e.eval_name for e in evaluators]}")

    # Use Braintrust's built-in create_app which handles all the setup
    # This creates a Starlette ASGI app with routes, middleware, etc.
    return create_app(evaluators, org_name=None)


# Optional: Add a local entrypoint for testing
@app.local_entrypoint()
def test():
    """Test the deployment locally."""
    print("Testing Braintrust eval server deployment...")
    print("Deploy with: modal deploy src/eval_server_simple.py")
    print("After deployment, you can connect to it from the Braintrust Playground")

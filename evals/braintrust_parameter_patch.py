"""
Patch for Braintrust SDK to properly handle Pydantic parameter models.

This fixes the parameters_to_json_schema() function to:
1. Detect single-field Pydantic models (e.g., models with just a 'value' field)
2. Extract the field's default value and description
3. Send a simple scalar schema instead of an object schema

This patch can be removed once Braintrust SDK fixes the TODO at:
braintrust/sdk/py/src/braintrust/parameters.py:149
"""

from typing import Any


def _pydantic_to_json_schema(model: Any) -> dict[str, Any]:
    """Convert a pydantic model to JSON schema."""
    if hasattr(model, "model_json_schema"):
        # pydantic 2
        return model.model_json_schema()
    elif hasattr(model, "schema"):
        # pydantic 1
        return model.schema()
    else:
        raise ValueError(
            f"Cannot convert {model} to JSON schema - not a pydantic model"
        )


def _get_pydantic_field_info(model_class: Any, field_name: str) -> dict[str, Any]:
    """Extract field information (default, description) from a Pydantic model."""
    result = {}

    # Try Pydantic v2 first
    if hasattr(model_class, "model_fields"):
        field_info = model_class.model_fields.get(field_name)
        if field_info:
            # Extract default
            if hasattr(field_info, "default") and field_info.default is not None:
                result["default"] = field_info.default
            elif hasattr(field_info, "default_factory") and field_info.default_factory:
                try:
                    result["default"] = field_info.default_factory()
                except Exception:
                    pass

            # Extract description
            if hasattr(field_info, "description") and field_info.description:
                result["description"] = field_info.description

    # Fallback to Pydantic v1
    elif hasattr(model_class, "__fields__"):
        field_info = model_class.__fields__.get(field_name)
        if field_info:
            # Extract default
            if hasattr(field_info, "default") and field_info.default is not None:
                result["default"] = field_info.default
            elif hasattr(field_info, "default_factory") and field_info.default_factory:
                try:
                    result["default"] = field_info.default_factory()
                except Exception:
                    pass

            # Extract description from field_info
            if hasattr(field_info, "field_info") and hasattr(
                field_info.field_info, "description"
            ):
                if field_info.field_info.description:
                    result["description"] = field_info.field_info.description

    return result


def patched_parameters_to_json_schema(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    Convert EvalParameters to JSON schema format for serialization.

    This is a PATCHED version that properly handles Pydantic models with single fields.
    """
    result = {}

    for name, schema in parameters.items():
        if isinstance(schema, dict) and schema.get("type") == "prompt":
            # Prompt parameter
            result[name] = {
                "type": "prompt",
                "default": schema.get("default"),
                "description": schema.get("description"),
            }
        else:
            # Pydantic model
            try:
                # Check if it's a single-field Pydantic model
                # (matches the logic in validate_parameters)
                fields = getattr(schema, "__fields__", None) or getattr(
                    schema, "model_fields", {}
                )

                if len(fields) == 1:
                    # Single-field model - extract the field's schema
                    field_name = list(fields.keys())[0]
                    full_schema = _pydantic_to_json_schema(schema)

                    # Extract just the field's schema (not the wrapper object)
                    if (
                        "properties" in full_schema
                        and field_name in full_schema["properties"]
                    ):
                        field_schema = full_schema["properties"][field_name]

                        # Extract default and description from the Pydantic field
                        field_info = _get_pydantic_field_info(schema, field_name)

                        # Merge field schema with extracted info
                        result[name] = {
                            "type": "data",
                            "schema": field_schema,
                        }

                        if "default" in field_info:
                            result[name]["default"] = field_info["default"]
                        if "description" in field_info:
                            result[name]["description"] = field_info["description"]
                    else:
                        # Fallback: use full schema
                        result[name] = {
                            "type": "data",
                            "schema": full_schema,
                        }
                else:
                    # Multi-field model - use full schema as-is
                    result[name] = {
                        "type": "data",
                        "schema": _pydantic_to_json_schema(schema),
                    }

            except (ValueError, AttributeError):
                # Not a pydantic model, skip
                pass

    return result


def apply_parameter_patch():
    """Apply the parameter patch to the Braintrust SDK.

    Note: Due to Python's import system, this patch must be applied BEFORE
    any Braintrust modules import parameters_to_json_schema. For local dev server,
    this is challenging because the dev server imports happen before our eval file loads.
    """
    try:
        import sys

        import braintrust.parameters as params_module  # pyright: ignore[reportMissingImports]

        # Replace the function in the module
        params_module.parameters_to_json_schema = patched_parameters_to_json_schema

        # CRITICAL: Also patch in any modules that already imported it
        # The dev server imports this before our patch runs
        for module_name, module in list(sys.modules.items()):
            if "braintrust" in module_name and hasattr(
                module, "parameters_to_json_schema"
            ):
                module.parameters_to_json_schema = patched_parameters_to_json_schema  # type: ignore
        return True

    except ImportError as e:
        print(f"⚠ Failed to apply Braintrust parameter patch: {e}")
        return False

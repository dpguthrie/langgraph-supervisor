# Braintrust Parameter Patch

## Overview

This patch fixes a limitation in the Braintrust SDK where Pydantic parameter models with default values don't work properly in the playground UI.

## The Problem

The Braintrust SDK's `parameters_to_json_schema()` function has a TODO comment (line 149 in `braintrust/parameters.py`) indicating that default value extraction from Pydantic models is not implemented. This causes:

1. Default values don't show up in the playground UI
2. The UI expects object/dict input instead of plain values
3. YAML validation errors: "must be object"

## The Solution

We've created a local patch that:

1. **Detects single-field Pydantic models** (models with just a `value` field)
2. **Extracts the field's schema** instead of wrapping it in an object
3. **Extracts default values and descriptions** from the Pydantic field
4. **Sends simple scalar schemas** to the UI (e.g., string, not `{value: string}`)

## Files Created/Modified

### New Files

- **`evals/braintrust_parameter_patch.py`**: Contains the patched `parameters_to_json_schema` function
  - `patched_parameters_to_json_schema()`: Fixed version that handles single-field models
  - `apply_parameter_patch()`: Monkey-patches the Braintrust SDK at runtime

### Modified Files

- **`evals/parameters.py`**: Parameter definitions using single-field pattern
  - All parameter classes now have a single `value` field
  - Includes default value and description

- **`evals/eval_supervisor.py`**: 
  - `unwrap_parameters()` extracts the `value` field from parameter instances
  - Parameters are enabled in the `Eval()` call

- **`src/eval_server.py`**: 
  - Applies the patch at server startup (before loading evaluators)
  - Ensures the patch is included in the Modal image

## How It Works

### 1. Parameter Definition
```python
class SystemPromptParam(BaseModel):
    value: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        description="Custom system prompt for the supervisor agent.",
    )
```

### 2. SDK Patch Applied
The patch modifies how the schema is sent to the UI:

**Before** (broken):
```json
{
  "type": "data",
  "schema": {
    "properties": {
      "value": {"type": "string", "default": "..."}
    },
    "required": ["value"]
  }
}
```

**After** (fixed):
```json
{
  "type": "data", 
  "schema": {"type": "string"},
  "default": "...",
  "description": "..."
}
```

### 3. Value Extraction
The `unwrap_parameters()` function extracts just the `.value` field when the eval runs.

## Testing

### Local Testing
```bash
# Run eval locally to verify parameters work
braintrust eval evals/eval_supervisor.py --no-send-logs
```

### Remote Testing (Modal)
```bash
# Deploy the eval server with the patch
modal deploy src/eval_server.py

# The patch will be applied automatically when the server starts
# Check the logs for: "✓ Applied Braintrust parameter patch"
```

### Verify in Playground
1. Open the Braintrust playground
2. Connect to your remote eval server
3. You should now see:
   - Default values pre-filled in the parameter fields
   - Ability to edit values as plain strings (not YAML objects)
   - No validation errors

## When to Remove This Patch

This patch can be removed once Braintrust fixes the TODO at:
`braintrust/sdk/py/src/braintrust/parameters.py:149`

To check if it's fixed:
1. Update the `braintrust` package
2. Comment out the patch application in `eval_server.py`
3. Test if parameters work in the playground
4. If yes, delete `braintrust_parameter_patch.py`

## Troubleshooting

### Patch Not Applied
Check the server logs for the patch message. If you see:
```
⚠ Failed to apply Braintrust parameter patch: ...
```

This means the Braintrust SDK structure has changed. You may need to update the patch.

### Still Seeing "must be object" Error
Ensure:
1. The patch is being applied (check logs)
2. Your parameters use the single-field `value` pattern
3. The Modal image includes the `evals/` directory

### Values Not Showing in UI
Verify:
1. Field defaults are set in the Pydantic models
2. The patch's `_get_pydantic_field_info()` can extract them
3. Check browser console for any JavaScript errors

"""Custom Braintrust tracing utilities for improved observability."""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from braintrust import SpanTypeAttribute
from braintrust_langchain import BraintrustCallbackHandler

logger = logging.getLogger(__name__)

# Set to True to enable detailed tracing debug logs
ENABLE_TRACE_LOGGING = False


class ImprovedBraintrustCallbackHandler(BraintrustCallbackHandler):
    """Enhanced BraintrustCallbackHandler with better span types and naming.

    This handler improves upon the base BraintrustCallbackHandler by:
    1. Setting proper TOOL span type for tool invocations (instead of defaulting to TASK)
    2. Improving subagent span naming for better trace readability
    3. Adding metadata to help distinguish different types of operations

    ## Extensibility & Design Principles

    This handler uses tag-based naming conventions to work with any subagent architecture:

    - **Subagent Detection**: Any chain/tool with tags matching `subagent:<name>` will be
      automatically prefixed with the subagent name

    - **Tool Naming**: Tools within subagents get prefixed as `<subagent>.<tool_name>`
      (e.g., "SQL Agent.execute_sql_query")

    - **Node Naming**: LangGraph nodes within subagents get prefixed as `<subagent>.<node_name>`
      (e.g., "SQL Agent.generate_query")

    - **Launcher Detection**: Tools named "task" with `subagent_type` in inputs are treated
      as subagent launchers and renamed to the subagent type

    To use with new subagent types, simply ensure your subagent wrapper adds appropriate
    `subagent:<name>` tags to the execution context, similar to `create_named_subagent_wrapper`
    in deep_agent.py.
    """

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Override to disable context tracking."""
        from braintrust_langchain.callbacks import last_item

        name = (
            name
            or serialized.get("name")
            or last_item(serialized.get("id") or [])
            or "LLM"
        )

        self._start_span(
            parent_run_id,
            run_id,
            name=name,
            type=SpanTypeAttribute.LLM,
            set_current=False,  # Disable context tracking to avoid cross-context errors
            event={
                "input": prompts,
                "tags": tags,
                "metadata": {
                    "serialized": serialized,
                    "name": name,
                    "metadata": metadata,
                    **kwargs,
                },
            },
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Override to disable context tracking."""
        from braintrust_langchain.callbacks import last_item

        invocation_params = invocation_params or {}

        self._start_span(
            parent_run_id,
            run_id,
            name=name
            or serialized.get("name")
            or last_item(serialized.get("id") or [])
            or "Chat Model",
            type=SpanTypeAttribute.LLM,
            set_current=False,  # Disable context tracking to avoid cross-context errors
            event={
                "input": messages,
                "tags": tags,
                "metadata": {
                    "serialized": serialized,
                    "invocation_params": invocation_params,
                    "metadata": metadata or {},
                    "name": name,
                    **kwargs,
                },
            },
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Override to set proper TOOL span type."""
        from braintrust_langchain.callbacks import last_item, safe_parse_serialized_json

        # Determine if this is a subagent tool call
        tool_name = (
            name
            or serialized.get("name")
            or last_item(serialized.get("id") or [])
            or "Tool"
        )

        # Determine the context from tags and metadata
        langgraph_node = metadata.get("langgraph_node") if metadata else None
        subagent_tags = [tag for tag in (tags or []) if tag.startswith("subagent:")]

        # Check if we're inside a subagent execution
        current_subagent = None
        if subagent_tags:
            current_subagent = subagent_tags[0].replace("subagent:", "")

        # Check if this tool IS a subagent launcher
        is_subagent_launcher = (
            tool_name == "task" and inputs is not None and "subagent_type" in inputs
        )

        if is_subagent_launcher:
            # This is the subagent launcher - name it after the subagent
            subagent_type = (
                inputs.get("subagent_type", "Unknown Agent")
                if inputs
                else "Unknown Agent"
            )
            tool_name = f"invoke_{subagent_type.lower().replace(' ', '_')}"
            span_type = SpanTypeAttribute.TASK
        elif current_subagent:
            # We're inside a subagent - prefix tool name with subagent
            tool_name = f"{current_subagent}.{tool_name}"
            span_type = SpanTypeAttribute.TOOL
        else:
            # Regular tool call
            span_type = SpanTypeAttribute.TOOL

        # Create serializable inputs - remove non-JSON-serializable objects like ToolRuntime
        safe_inputs = None
        if inputs:
            safe_inputs = {}
            for k, v in inputs.items():
                # Skip runtime objects and other non-serializable items
                if (
                    k == "runtime"
                    or hasattr(v, "__class__")
                    and "Runtime" in v.__class__.__name__
                ):
                    safe_inputs[k] = f"<{type(v).__name__}>"
                else:
                    try:
                        # Try to serialize to verify it's JSON-safe
                        import json

                        json.dumps(v)
                        safe_inputs[k] = v
                    except (TypeError, ValueError):
                        safe_inputs[k] = str(v)[
                            :200
                        ]  # Truncate long string representations

        self._start_span(
            parent_run_id,
            run_id,
            name=tool_name,
            type=span_type,
            set_current=False,  # Disable context tracking to avoid cross-context errors
            event={
                "input": safe_inputs or safe_parse_serialized_json(input_str),
                "tags": tags,
                "metadata": {
                    "metadata": metadata,
                    "serialized": serialized,
                    "input_str": input_str[:500],  # Truncate long input strings
                    "name": name,
                    "is_subagent_launcher": is_subagent_launcher,
                    "current_subagent": current_subagent,
                    "langgraph_node": langgraph_node,
                },
            },
        )

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Override to improve naming for LangGraph nodes and subagents."""
        from braintrust_langchain.callbacks import last_item

        tags = tags or []

        # avoids extra logs that seem not as useful esp. with langgraph
        if "langsmith:hidden" in tags:
            self.skipped_runs.add(run_id)
            return

        metadata = metadata or {}

        # Improved name resolution
        langgraph_node = metadata.get("langgraph_node")

        # Check if this is a subagent-related node
        subagent_tags = [tag for tag in tags if tag.startswith("subagent:")]
        is_subagent_node = bool(subagent_tags)

        # Build a more descriptive name
        if is_subagent_node:
            # Extract subagent name from tags
            subagent_name = subagent_tags[0].replace("subagent:", "")

            # Check if this is the wrapper function (invoke_with_name)
            if name == "invoke_with_name":
                resolved_name = f"→ {subagent_name}"
            # If this is a node within a subagent graph, prefix with subagent name
            elif langgraph_node and langgraph_node != "tools":
                resolved_name = f"{subagent_name}.{langgraph_node}"
            else:
                resolved_name = subagent_name
        else:
            # Check if this is the wrapper function before subagent tags are applied
            if name == "invoke_with_name":
                resolved_name = "invoke_subagent"
            else:
                resolved_name = (
                    langgraph_node
                    or name
                    or serialized.get("name")
                    or last_item(serialized.get("id") or [])
                    or "Chain"
                )

                # Replace generic "LangGraph" name with something more descriptive
                if resolved_name == "LangGraph" or resolved_name == "RunnableSequence":
                    # Check if this is the top-level deep agent (no parent)
                    if parent_run_id is None:
                        resolved_name = "Supervisor Agent"
                    else:
                        # For nested LangGraph chains, use a more generic name
                        resolved_name = "Agent Graph"

        self._start_span(
            parent_run_id,
            run_id,
            name=resolved_name,
            set_current=False,  # Disable context tracking to avoid cross-context errors
            event={
                "input": inputs,
                "tags": tags,
                "metadata": {
                    "serialized": serialized,
                    "name": name,
                    "metadata": metadata,
                    "is_subagent_node": is_subagent_node,
                    "langgraph_node": langgraph_node,
                    **kwargs,
                },
            },
        )

    def on_chain_end(
        self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs: Any
    ) -> Any:
        """Override to handle skipped runs."""
        if run_id in self.skipped_runs:
            return
        return super().on_chain_end(outputs, run_id=run_id, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> Any:
        """Override to handle skipped runs."""
        if run_id in self.skipped_runs:
            return
        return super().on_llm_end(response, run_id=run_id, **kwargs)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> Any:
        """Override to handle skipped runs."""
        if run_id in self.skipped_runs:
            return
        return super().on_tool_end(output, run_id=run_id, **kwargs)

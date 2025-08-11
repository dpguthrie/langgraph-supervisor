import os

from braintrust import Attachment, init_logger
from braintrust_langchain import BraintrustCallbackHandler, set_global_handler
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

load_dotenv()


class GraphAwareBraintrustCallbackHandler(BraintrustCallbackHandler):
    """Braintrust callback handler that includes graph metadata in root spans."""

    def __init__(self, graph_metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.graph_metadata = graph_metadata or {}

    def _start_span(
        self,
        parent_run_id,
        run_id,
        name=None,
        type=None,
        span_attributes=None,
        start_time=None,
        set_current=None,
        parent=None,
        event=None,
    ):
        # If this is a root span (no parent_run_id), include graph metadata
        if not parent_run_id and self.graph_metadata:
            if event is None:
                event = {}

            # Add graph metadata to the event
            if "metadata" not in event:
                event["metadata"] = {}

            # Update metadata with graph information
            event_metadata = event["metadata"]
            if isinstance(event_metadata, dict):
                event_metadata.update(self.graph_metadata)

        return super()._start_span(
            parent_run_id,
            run_id,
            name=name,
            type=type,
            span_attributes=span_attributes,
            start_time=start_time,
            set_current=set_current,
            parent=parent,
            event=event,  # type: ignore
        )


def _generate_graph_metadata(graph):
    """Generate graph visualization and metadata for Braintrust traces."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        attachment = Attachment(
            data=png_data,
            filename="mermaid_diagram.png",
            content_type="image/png",
        )
        return {"graph_attachment": attachment}
    except Exception as e:
        # Non-fatal: fall back to structured metadata
        return {
            "graph_structure": "supervisor_with_research_and_math_agents",
            "graph_error": str(e),
        }


_cached_supervisor = None


def _build_supervisor():
    init_logger(
        project="langgraph-supervisor", api_key=os.environ.get("BRAINTRUST_API_KEY")
    )

    web_search = TavilySearch(max_results=3)

    research_agent = create_react_agent(
        model="openai:gpt-4.1",
        tools=[web_search],
        prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any math\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_agent",
    )

    def add(a: float, b: float):
        """Add two numbers and return their sum."""
        return a + b

    def subtract(a: float, b: float):
        """Subtract b from a and return the result."""
        return a - b

    def multiply(a: float, b: float):
        """Multiply two numbers and return the product."""
        return a * b

    def divide(a: float, b: float):
        """Divide a by b and return the quotient. Raises if b is zero."""
        return a / b

    math_agent = create_react_agent(
        model="openai:gpt-4.1",
        tools=[add, subtract, multiply, divide],
        prompt=(
            "You are a math agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with math-related tasks\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="math_agent",
    )

    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-4.1"),
        agents=[research_agent, math_agent],
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- a research agent. Assign research-related tasks to this agent\n"
            "- a math agent. Assign math-related tasks to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()

    # Set up Braintrust callback with graph metadata
    graph_metadata = _generate_graph_metadata(supervisor)
    handler = GraphAwareBraintrustCallbackHandler(graph_metadata=graph_metadata)
    set_global_handler(handler)

    return supervisor


def get_supervisor(force_rebuild: bool = False):
    global _cached_supervisor
    if force_rebuild or _cached_supervisor is None:
        _cached_supervisor = _build_supervisor()
    return _cached_supervisor

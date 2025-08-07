import getpass
import os

from braintrust import Attachment, init_logger
from braintrust_langchain import BraintrustCallbackHandler, set_global_handler
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from helpers import pretty_print_messages

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


def generate_graph_metadata(graph):
    """Generate graph visualization and metadata for Braintrust traces."""
    try:
        # Generate PNG image
        png_data = graph.get_graph().draw_mermaid_png()

        attachment = Attachment(
            data=png_data,
            filename="mermaid_diagram.png",
            content_type="image/png",
        )

        return {"graph_attachment": attachment}

    except Exception as e:
        print(f"Warning: Could not generate graph visualization: {e}")
        return {
            "graph_structure": "supervisor_with_research_and_math_agents",
            "graph_error": str(e),
        }


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")
_set_if_undefined("BRAINTRUST_API_KEY")

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
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float):
    """Subtract two numbers."""
    return a - b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
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

# Generate graph metadata at startup
graph_metadata = generate_graph_metadata(supervisor)

# Create custom handler with graph metadata
handler = GraphAwareBraintrustCallbackHandler(graph_metadata=graph_metadata)
set_global_handler(handler)

if __name__ == "__main__":
    console = Console()

    # Welcome message
    welcome_text = Text("🤖 LangGraph Supervisor Chat", style="bold cyan")
    welcome_panel = Panel(
        welcome_text, subtitle="Type 'quit' or 'q' to exit", border_style="cyan"
    )
    console.print(welcome_panel)
    console.print()

    history = []

    while True:
        try:
            # Get user input with Rich prompt
            user_input = Prompt.ask("[bold green]You[/bold green]", console=console)

            if user_input.lower() in {"q", "quit", "exit"}:
                console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
                break

            if not user_input.strip():
                continue

            history = [HumanMessage(content=user_input)]

            # Show processing indicator
            with console.status("[bold blue]Processing...", spinner="dots"):
                pass  # Status will auto-stop when exiting context

            for event in supervisor.stream({"messages": history}):
                pretty_print_messages(event)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")

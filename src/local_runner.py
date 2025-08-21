import getpass
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.helpers import pretty_print_messages


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


def main():
    load_dotenv()

    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")
    _set_if_undefined("BRAINTRUST_API_KEY")

    console = Console()

    # Import after environment is set so agent initialization has keys available
    from src.agent_graph import get_supervisor

    supervisor = get_supervisor()

    welcome_text = Text("🤖 LangGraph Supervisor Chat", style="bold cyan")
    welcome_panel = Panel(
        welcome_text, subtitle="Type 'quit' or 'q' to exit", border_style="cyan"
    )
    console.print(welcome_panel)
    console.print()

    history = []

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]", console=console)

            if user_input.lower() in {"q", "quit", "exit"}:
                console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
                break

            if not user_input.strip():
                continue

            history = [HumanMessage(content=user_input)]

            with console.status("[bold blue]Processing...", spinner="dots"):
                pass

            for event in supervisor.stream({"messages": history}):
                pretty_print_messages(event)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]👋 Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


if __name__ == "__main__":
    main()

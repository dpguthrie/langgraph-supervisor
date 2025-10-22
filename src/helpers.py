from langchain_core.messages import convert_to_messages
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def pretty_print_message(message, indent=False):
    """Print a single message with Rich formatting."""
    content = message.content
    if not content:
        content = message.tool_calls[0]["name"]
    message_type = message.__class__.__name__

    # print(message)

    # Color code by message type
    if "Human" in message_type:
        style = "bold green"
        icon = "👤"
    elif "AI" in message_type or "Assistant" in message_type:
        style = "bold blue"
        icon = "🤖"
    elif "Tool" in message_type:
        style = "bold yellow"
        icon = "🔧"
    else:
        style = "bold white"
        icon = "💬"

    # Create formatted message
    title = f"{icon} {message_type}"

    if indent:
        # For subgraph messages, use a simpler format
        console.print(f"    [dim]{title}:[/dim] {content}")
    else:
        # Main messages get a panel
        panel = Panel(
            content,
            title=title,
            title_align="left",
            border_style=style.split()[-1],  # Extract color from style
            padding=(0, 1),
        )
        console.print(panel)


def pretty_print_messages(update, last_message=False):
    """Print messages from supervisor updates with Rich formatting."""
    is_subgraph = False

    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        subgraph_text = Text(f"📊 Subgraph: {graph_id}", style="bold magenta")
        console.print(subgraph_text)
        is_subgraph = True

    for node_name, node_update in update.items():
        # Skip middleware events that don't have message updates
        if node_update is None or not isinstance(node_update, dict):
            continue

        # Skip if no messages in this update
        if "messages" not in node_update:
            continue

        # Create a header for each node update
        if is_subgraph:
            node_text = Text(f"  ⚡ Node: {node_name}", style="dim cyan")
        else:
            node_text = Text(f"⚡ Agent: {node_name}", style="bold cyan")

        console.print(node_text)

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)

        console.print()  # Add spacing between updates

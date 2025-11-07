#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore

from braintrust import init_logger
from braintrust_langchain import set_global_handler, BraintrustCallbackHandler


# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()


def generate_questions(num_questions: int, seed: Optional[int] = None) -> List[str]:
    """Generate diverse questions mixing math-only, research-only, and hybrid questions.

    Questions test the supervisor's ability to:
    - Route pure math questions to MATH_AGENT
    - Route pure research questions to RESEARCH_AGENT
    - Coordinate BOTH agents for hybrid questions (research + math)
    """
    rng = random.Random(seed)

    # Split: ~1/3 math, ~1/3 research, ~1/3 hybrid
    num_math = num_questions // 3
    num_research = num_questions // 3
    num_hybrid = num_questions - num_math - num_research

    print(
        f"Generating {num_math} math, {num_research} research, "
        f"and {num_hybrid} hybrid questions..."
    )

    model = init_chat_model("openai:gpt-4.1", temperature=0.9)

    prompt = (
        f"Generate exactly {num_questions} diverse questions:\n\n"
        f"1. {num_math} PURE MATH questions (arithmetic, algebra, percentages, calculations)\n"
        f"2. {num_research} PURE RESEARCH questions (history, geography, current events, facts)\n"
        f"3. {num_hybrid} HYBRID questions that combine research + math\n\n"
        "HYBRID examples:\n"
        "- 'What year was the Eiffel Tower built? Multiply that by 2 and tell me if it's greater than 4000.'\n"
        "- 'How many states are in the USA? Add 13 to that number.'\n"
        "- 'What's the population of Tokyo? Is it more than 10 million times 2?'\n\n"
        "Make questions diverse and interesting.\n"
        "Return ONLY a JSON array of strings, no commentary.\n"
        "Keep each question under 150 characters.\n"
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    text = getattr(resp, "content", "") or ""
    questions = json.loads(text)

    if not (isinstance(questions, list) and all(isinstance(q, str) for q in questions)):
        raise RuntimeError("LLM did not return a valid JSON array of strings")

    # Shuffle for variety
    rng.shuffle(questions)

    print(f"✓ Generated {len(questions)} total questions (shuffled)")
    return questions[:num_questions]


async def run_question(supervisor, question: str) -> tuple[str, bool, Optional[dict]]:
    """Run a question through the supervisor.

    Returns:
        Tuple of (question, success, result)
    """
    try:
        print(f"📥 Running: {question}")
        result = await supervisor.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        messages = result.get("messages", []) if isinstance(result, dict) else []

        # Extract final response
        if messages:
            final_msg = messages[-1]
            content = getattr(final_msg, "content", str(final_msg))
            print(f"✅ Response: {content[:100]}...")

        return question, True, result
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return question, False, None


async def main_async(args):
    """Run questions through the supervisor concurrently."""
    # Check required environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY in environment", file=sys.stderr)
        sys.exit(2)

    # Import supervisor after setting up path
    from src.agent_graph import get_supervisor  # noqa: E402

    num_questions = random.randint(1, 100)
    print(f"Generating {num_questions} questions...\n")

    questions = generate_questions(num_questions, args.seed)
    print(f"Generated {len(questions)} questions\n")

    # Initialize supervisor once (reused for all questions)
    print("Initializing supervisor...")
    supervisor = get_supervisor()
    print("Supervisor ready!\n")

    # Run questions concurrently
    print(f"{'=' * 80}")
    print(f"Running {len(questions)} questions with concurrency={args.concurrency}")
    print(f"{'=' * 80}\n")

    successes = 0
    failures = 0

    # Process in batches to limit concurrency
    for i in range(0, len(questions), args.concurrency):
        batch = questions[i : i + args.concurrency]
        tasks = [run_question(supervisor, q) for q in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                failures += 1
            elif isinstance(result, tuple) and len(result) == 3:
                _, success, _ = result
                if success:
                    successes += 1
                else:
                    failures += 1
            else:
                failures += 1

        print()  # Blank line between batches

    print(f"\n{'=' * 80}")
    print(f"Completed. successes={successes} failures={failures}")
    print(f"{'=' * 80}\n")

    if args.fail_on_error and failures > 0:
        sys.exit(1)


def main(logger=None):
    parser = argparse.ArgumentParser(
        description="Generate N questions and run through supervisor locally"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "3")),
        help="Number of concurrent questions to process (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any request fails",
    )
    args = parser.parse_args()

    # Initialize tracing - set global handler BEFORE creating agents
    if logger is None:
        logger = init_logger(
            project="langgraph-supervisor", api_key=os.environ.get("BRAINTRUST_API_KEY")
        )
    set_global_handler(BraintrustCallbackHandler(logger=logger))

    # Run async main
    try:
        asyncio.run(main_async(args))
    finally:
        # Flush logger to ensure traces are sent to Braintrust
        print("\nFlushing traces to Braintrust...")
        logger.flush()
        print("✅ Traces sent!")


if __name__ == "__main__":
    main()

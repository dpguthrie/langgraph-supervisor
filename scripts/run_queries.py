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

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

# Import supervisor after setting up path
from src.agent_graph import get_supervisor  # noqa: E402


def generate_questions(num_questions: int, seed: Optional[int] = None) -> List[str]:
    """Generate a mix of math and research questions to test supervisor routing.

    The questions are designed to test the supervisor's ability to route to:
    - MATH_AGENT: arithmetic, calculations, equations
    - RESEARCH_AGENT: factual questions, current events, history, etc.
    """
    rng = random.Random(seed)

    # Calculate split (roughly 50/50 math vs research, with some variation)
    num_math = rng.randint(num_questions // 3, 2 * num_questions // 3)
    num_research = num_questions - num_math

    print(
        f"Generating {num_math} math questions and {num_research} research questions..."
    )

    model = init_chat_model("openai:gpt-4.1", temperature=0.9)

    # Generate math questions
    math_prompt = (
        f"Generate exactly {num_math} diverse mathematical questions.\n"
        "Include:\n"
        "- Basic arithmetic (addition, subtraction, multiplication, division)\n"
        "- Percentages and fractions\n"
        "- Simple algebra\n"
        "- Word problems with numbers\n"
        "- Calculations and conversions\n"
        "Make each question unique and varied in difficulty.\n"
        "Return ONLY a JSON array of strings, no commentary.\n"
        "Keep each question under 120 characters.\n"
    )
    math_resp = model.invoke([HumanMessage(content=math_prompt)])
    math_text = getattr(math_resp, "content", "") or ""
    math_questions = json.loads(math_text)

    # Generate research questions
    research_topics = [
        "history",
        "geography",
        "current events",
        "science facts",
        "famous people",
        "countries and capitals",
        "animals and nature",
        "technology companies",
        "literature",
        "movies and entertainment",
    ]
    selected_topics = rng.sample(research_topics, min(5, len(research_topics)))

    research_prompt = (
        f"Generate exactly {num_research} factual research questions about: {', '.join(selected_topics)}.\n"
        "Questions should:\n"
        "- Ask 'who', 'what', 'where', 'when' about real-world facts\n"
        "- Cover current events, history, geography, science\n"
        "- NOT involve calculations or math\n"
        "- Require looking up information or knowledge\n"
        "Make each question unique and interesting.\n"
        "Return ONLY a JSON array of strings, no commentary.\n"
        "Keep each question under 120 characters.\n"
    )
    research_resp = model.invoke([HumanMessage(content=research_prompt)])
    research_text = getattr(research_resp, "content", "") or ""
    research_questions = json.loads(research_text)

    # Validate and combine
    if not (
        isinstance(math_questions, list)
        and all(isinstance(q, str) for q in math_questions)
    ):
        raise RuntimeError("LLM did not return valid math questions")
    if not (
        isinstance(research_questions, list)
        and all(isinstance(q, str) for q in research_questions)
    ):
        raise RuntimeError("LLM did not return valid research questions")

    # Intersperse the questions randomly
    all_questions = math_questions[:num_math] + research_questions[:num_research]
    rng.shuffle(all_questions)

    print(f"✓ Generated {len(all_questions)} total questions (shuffled)")
    return all_questions


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


def main():
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

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

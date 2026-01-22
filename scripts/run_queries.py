#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

from braintrust import init_logger
from braintrust_langchain import BraintrustCallbackHandler, set_global_handler
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore

from src.config import AgentConfig

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

# Model pool for random selection per question
MODEL_POOL = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]


def generate_questions(num_questions: int, seed: Optional[int] = None) -> List[str]:
    """Generate realistic, varied questions with natural language variation and edge cases.

    Questions test the supervisor's ability to:
    - Route pure math questions to MATH_AGENT
    - Route pure research questions to RESEARCH_AGENT
    - Coordinate BOTH agents for hybrid questions (research + math)
    - Handle edge cases, ambiguous questions, and frustrated users
    """
    rng = random.Random(seed)

    print(f"Generating {num_questions} realistic questions with natural variation...")

    model = init_chat_model("openai:gpt-4o-mini", temperature=1.0)

    prompt = f"""Generate exactly {num_questions} realistic user questions that test an AI agent system.

Create a DIVERSE MIX of:

QUESTION TYPES (~distribute naturally, don't be exact):
- Pure MATH questions (calculations, arithmetic, algebra)
- Pure RESEARCH questions (facts, history, geography, current events)
- HYBRID questions (research + math combined)
- EDGE CASES (general chat, ambiguous, doesn't need specialized agents)

TONE & STYLE VARIATION (make it feel like real users):
- ~60% Normal, well-formed questions
- ~20% Informal/casual: typos, "u" instead of "you", "btw", lowercase, abbreviations
- ~10% FRUSTRATED users: complaints about accuracy, speed, repetition, capability
- ~10% Overly complex, rambling, or unclear questions

FRUSTRATED USER EXAMPLES:
- "I ALREADY asked this can you just answer??"
- "are u sure thats right? sounds wrong"
- "this is taking forever just give me the answer"
- "can you even do basic math or what"
- "why r u making this so complicated"

INFORMAL EXAMPLES:
- "whats 45 * 67 btw"
- "how many ppl live in tokyo"
- "tell me bout the french revolution"

AMBIGUOUS/EDGE EXAMPLES:
- "tell me about Paris" (unclear what aspect)
- "hello" (general chat)
- "help me understand this" (unclear what "this" is)

HYBRID EXAMPLES (research + meaningful calculation):
- "What's Tesla's current stock price? How much would 100 shares cost?"
- "How much taller is Mount Everest than K2 in meters?"
- "What's the distance from NYC to LA and how long to drive at 65mph?"
- "Apple's revenue in 2023 - what would a 15% increase be?"
- "If it's 3pm in Tokyo right now, what time is it in London?"
- "What percentage of California's population lives in Los Angeles?"
- "How many years between the first iPhone and first Android phone launch?"

CRITICAL OUTPUT FORMAT:
You MUST respond with ONLY a valid JSON array of strings. No markdown, no code blocks, no explanation.
Format example: ["question 1", "question 2", "question 3"]

REQUIREMENTS:
- Keep questions under 200 characters
- Make it feel REAL - include typos, emotions, varying complexity
- Don't make them all perfect or polite
- Return ONLY the JSON array, nothing else"""

    resp = model.invoke([HumanMessage(content=prompt)])
    text = getattr(resp, "content", "") or ""

    # Strip markdown code blocks if present
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code block markers
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()

    try:
        questions = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON. LLM response was:\n{text[:500]}")
        raise RuntimeError(f"LLM did not return valid JSON: {e}") from e

    if not (isinstance(questions, list) and all(isinstance(q, str) for q in questions)):
        raise RuntimeError("LLM did not return a valid JSON array of strings")

    # Shuffle for variety
    rng.shuffle(questions)

    print(f"✓ Generated {len(questions)} realistic questions (shuffled)")
    return questions[:num_questions]


async def run_question(question: str) -> tuple[str, bool, Optional[dict]]:
    """Run a question through the supervisor with a random model.

    Returns:
        Tuple of (question, success, result)
    """
    try:
        # Import supervisor getter inside function
        from src.agent_graph import get_supervisor  # noqa: E402

        # Randomly select model for this question
        selected_model = random.choice(MODEL_POOL)
        print(f"🎲 Using model: {selected_model}")
        print(f"📥 Running: {question}")

        # Create config with selected model for all agents
        config = AgentConfig(
            supervisor_model=selected_model,
            research_model=selected_model,
            math_model=selected_model,
        )

        # Get supervisor with this config (builds fresh, no caching)
        supervisor = get_supervisor(config)

        result = await supervisor.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            metadata={"customer_id": f"customer_{random.randint(1000, 9999)}"},
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

    # Run questions concurrently (each question gets its own supervisor with random model)
    print(f"{'=' * 80}")
    print(f"Running {len(questions)} questions with concurrency={args.concurrency}")
    print(f"Random model selected per question from: {', '.join(MODEL_POOL)}")
    print(f"{'=' * 80}\n")

    successes = 0
    failures = 0

    # Process in batches to limit concurrency
    for i in range(0, len(questions), args.concurrency):
        batch = questions[i : i + args.concurrency]
        tasks = [run_question(q) for q in batch]
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

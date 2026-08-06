#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import random
import re
import sys
from typing import List, Optional

from braintrust import init_logger
from braintrust_langchain import BraintrustCallbackHandler
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore

from src.config import AgentConfig

load_dotenv()

# Model pool for random selection per question
MODEL_POOL = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
QUESTION_CHAR_LIMIT = 300
GENERATION_ATTEMPTS = 2


TOO_EASY_PATTERNS = [
    re.compile(
        r"^\s*(what'?s|what is|calculate|compute)?\s*[\d\s+\-*/().,%]+[?!.]?\s*$", re.I
    ),
    re.compile(
        r"^\s*(add|subtract|multiply|divide)\s+\d+(\s+\w+){0,4}\s+\d+[?!.]?\s*$", re.I
    ),
    re.compile(r"\bwhat is the capital of\b", re.I),
    re.compile(r"\bwho is the president of\b", re.I),
    re.compile(r"\bwho won the fifa world cup in \d{4}\b", re.I),
    re.compile(r"\bwhy is water important\b", re.I),
    re.compile(r"\bwhy do(es)? the seasons change\b", re.I),
]


def looks_too_easy(question: str) -> bool:
    """Identify generated questions that are unlikely to expose agent failures."""
    normalized = " ".join(question.strip().split())
    lower = normalized.lower()

    if len(normalized) < 25:
        return True

    if any(pattern.search(normalized) for pattern in TOO_EASY_PATTERNS):
        return True

    easy_research_stems = (
        "tell me about ",
        "what is ",
        "who is ",
        "when was ",
        "why is ",
        "how far is ",
        "how many ",
    )
    hard_markers = (
        "current",
        "latest",
        "as of",
        "today",
        "source",
        "cite",
        "compare",
        "difference",
        "percent",
        "percentage",
        "growth",
        "revenue",
        "population",
        "market cap",
        "stock",
        "timezone",
        "utc",
        "which one",
        "those",
        "this",
        "not sure",
        "i meant",
    )
    if lower.startswith(easy_research_stems) and not any(
        marker in lower for marker in hard_markers
    ):
        return True

    return False


def dedupe_questions(questions: list[str]) -> list[str]:
    """Keep unique, non-empty questions while preserving model output order."""
    seen = set()
    unique_questions = []
    for question in questions:
        normalized = " ".join(question.strip().split())
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_questions.append(normalized)
    return unique_questions


def extract_questions(text: str) -> list[str]:
    """Parse the model's JSON array response into a list of strings."""
    text = text.strip()
    if text.startswith("```"):
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

    return questions


def build_generation_prompt(
    num_questions: int, rejected_questions: Optional[list[str]] = None
) -> str:
    """Build a prompt that favors failure-revealing user questions."""
    rejection_note = ""
    if rejected_questions:
        rejected_sample = json.dumps(rejected_questions[:12], ensure_ascii=True)
        rejection_note = f"""
The previous batch included questions that were too basic. Do NOT generate close
variants of these rejected examples: {rejected_sample}
"""

    return f"""Generate exactly {num_questions} realistic user questions that stress-test an AI agent supervisor.

The supervisor can either answer directly or route to:
- MathAgent: calculations only, once all numeric inputs are known.
- ResearchAgent: web/current/factual lookup and source gathering.

Recent generated logs were too easy: examples included "what's 15% of 250?",
"why is water important for life?", "what's the historical significance of the
Berlin Wall?", "how many continents are there?", and generic Mars distance
questions. Avoid that style. These questions should expose real failure modes,
not just prove that routing works for obvious single-intent prompts.

HARDNESS BAR:
- Every question must be meaningfully harder than a one-hop fact or arithmetic prompt.
- Every question must contain at least one concrete failure mode.
- Prefer questions that would reveal whether the agent can decide when to search,
  when to calculate, when to do both in order, when to answer directly, and when
  to ask a clarifying question.
- Do not force an even category mix. Difficulty matters more than coverage quotas.

FAILURE MODES TO USE:
- stale-knowledge traps: "current", "latest", "as of today", role changes, recent results.
- source constraints: ask for official/primary sources, citations, or reconcile disagreement.
- route-order traps: user asks for math using values that must be researched first.
- missing inputs: "those two cities", "that company", "the last one", or vague pronouns with no context.
- entity ambiguity: Paris Texas vs Paris France, Georgia country vs state, Apple stock vs fruit.
- correction turns in one prompt: "wait no, use revenue not profit" or "I meant metro area".
- unit/date traps: fiscal vs calendar year, local time/date, metric vs imperial, per-capita vs total.
- over-routing traps: tasks that look factual but are really editing, summarization, tone rewrite, or categorization.
- frustrated or informal wording where the difficulty is task ambiguity, not just rudeness.
- multi-step pure math, but only when it requires expression parsing, algebra, date/time, units, or careful order of operations.

GOOD EXAMPLES:
- "Use official sources: compare Apple's latest annual revenue with Microsoft's and tell me the percent gap."
- "I need the current mayors of Portland and Paris, but I mean Portland Maine; who's been in office longer?"
- "Wait no, not profit - use 2023 revenue for Meta and Google and calculate growth to the latest full year."
- "Can you make this sound less defensive: 'we missed the SLA because support dropped it'?"
- "How much would 37 shares of NVDA cost at the latest price? cite where the price came from."
- "Those two cities have similar names - compare population of Paris, TX and Paris, France. Use city proper."
- "I forgot to paste the numbers, but can you calculate the percent change between them?"
- "It's 9:40pm in Denver on June 16, 2026; what time/date is it in Tokyo?"

BAD EXAMPLES TO AVOID:
- "What is 23 + 47?"
- "What is the capital of Brazil?"
- "Who is the president of France?"
- "Tell me about the French Revolution."
- "Why is water important?"
- "What is the distance from Earth to Mars?"

STYLE:
- Make them sound like real users, including casual phrasing or typos where natural.
- Keep each question under {QUESTION_CHAR_LIMIT} characters.
- Every question should have a reason it could fail: wrong route, stale fact, missing clarification, bad source, wrong units, or unnecessary tool use.
- Do not include expected answers or labels.
{rejection_note}
CRITICAL OUTPUT FORMAT:
Return ONLY a valid JSON array of strings. No markdown, no code blocks, no explanation.
Format example: ["question 1", "question 2", "question 3"]"""


def generate_questions(num_questions: int, seed: Optional[int] = None) -> List[str]:
    """Generate challenging questions with natural language variation and edge cases.

    Questions test the supervisor's ability to:
    - Route pure math questions to MathAgent only when inputs are known
    - Route research questions to ResearchAgent when facts need verification
    - Coordinate both agents for research-first calculations
    - Avoid tool use for direct-answer tasks
    - Ask clarifying questions instead of guessing when inputs are missing
    """
    rng = random.Random(seed)

    print(f"Generating {num_questions} challenging questions with natural variation...")

    model = init_chat_model("openai:gpt-4o-mini", temperature=1.0)
    all_questions: list[str] = []
    rejected_questions: list[str] = []

    for _ in range(GENERATION_ATTEMPTS):
        prompt = build_generation_prompt(num_questions, rejected_questions)
        resp = model.invoke([HumanMessage(content=prompt)])
        text = getattr(resp, "content", "") or ""
        all_questions.extend(extract_questions(text))

        unique_questions = dedupe_questions(all_questions)
        hard_questions = [q for q in unique_questions if not looks_too_easy(q)]
        rejected_questions = [q for q in unique_questions if looks_too_easy(q)]
        if len(hard_questions) >= num_questions:
            break

    questions = dedupe_questions(all_questions)
    hard_questions = [q for q in questions if not looks_too_easy(q)]
    easy_questions = [q for q in questions if looks_too_easy(q)]

    # Prefer harder questions, but keep easy ones as a fallback to avoid underfilling.
    rng.shuffle(hard_questions)
    rng.shuffle(easy_questions)
    questions = hard_questions + easy_questions

    if easy_questions:
        print(f"Filtered {len(easy_questions)} basic question(s) to the backfill pool")
    print(
        f"✓ Generated {len(questions)} challenging questions (hard prompts prioritized)"
    )
    return questions[:num_questions]


async def run_question(question: str, logger) -> tuple[str, bool, Optional[dict]]:
    """Run a question through the supervisor with a random model.

    Returns:
        Tuple of (question, success, result)
    """
    try:
        # Import supervisor getter inside function
        from src.agents.deep_agent import get_supervisor  # noqa: E402

        # Randomly select model for this question
        selected_model = random.choice(MODEL_POOL)
        print(f"🎲 Using model: {selected_model}")
        print(f"📥 Running: {question}")

        # Create config with selected model for all agents
        agent_config = AgentConfig(
            supervisor_model=selected_model,
            research_model=selected_model,
            math_model=selected_model,
        )

        # Get supervisor with this config (builds fresh, no caching)
        supervisor = get_supervisor(agent_config)

        # One callback handler per invocation. The handler holds per-run span
        # state (self.spans, self.root_run_id) and sets contextvars, so sharing
        # a single instance across concurrent runs crosses their span trees and
        # loses writes. Metadata must go inside `config` -- LangGraph ignores it
        # as a bare ainvoke kwarg.
        result = await supervisor.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "callbacks": [BraintrustCallbackHandler(logger=logger)],
                "metadata": {"customer_id": f"customer_{random.randint(1000, 9999)}"},
            },
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


async def main_async(args, logger):
    """Run questions through the supervisor concurrently."""
    # Check required environment variables
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("Missing BRAINTRUST_API_KEY in environment", file=sys.stderr)
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
        tasks = [run_question(q, logger) for q in batch]
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

    # Initialize tracing. Each question builds its own callback handler in
    # run_question rather than registering one globally.
    if logger is None:
        logger = init_logger(
            project=os.environ["BRAINTRUST_PROJECT_NAME"],
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
        )

    # Run async main
    try:
        asyncio.run(main_async(args, logger))
    finally:
        # Flush logger to ensure traces are sent to Braintrust
        print("\nFlushing traces to Braintrust...")
        logger.flush()
        print("✅ Traces sent!")


if __name__ == "__main__":
    main()

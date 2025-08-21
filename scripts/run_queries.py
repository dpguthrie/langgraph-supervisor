#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model  # type: ignore
from langchain_core.messages import HumanMessage  # type: ignore

load_dotenv()


def generate_questions(num_questions: int, seed: Optional[int] = None) -> List[str]:
    """Generate questions using an LLM (OpenAI via LangChain)."""
    if seed is None:
        seed = int(time.time())
    model = init_chat_model("openai:gpt-4.1")
    prompt = (
        "Generate exactly "
        f"{num_questions}"
        " diverse user questions mixing math, research, current events, and general knowledge.\n"
        "Return ONLY a JSON array of strings, no commentary.\n"
        "Keep each question under 120 characters.\n"
        f"Seed: {seed}"
    )
    msg = HumanMessage(content=prompt)
    resp = model.invoke([msg])
    text = getattr(resp, "content", "") or ""
    questions = json.loads(text)
    if not (isinstance(questions, list) and all(isinstance(q, str) for q in questions)):
        raise RuntimeError("LLM did not return a valid JSON array of strings")
    return questions[:num_questions]


def post_question(
    session: requests.Session,
    url: str,
    modal_key: str,
    modal_secret: str,
    question: str,
    timeout_s: float = 30.0,
) -> Tuple[str, int, Optional[dict]]:
    headers = {
        "Modal-Key": modal_key,
        "Modal-Secret": modal_secret,
        "Content-Type": "application/json",
    }
    payload = {"q": question}
    try:
        resp = session.post(url, headers=headers, json=payload, timeout=timeout_s)
        return (
            question,
            resp.status_code,
            (
                resp.json()
                if resp.headers.get("Content-Type", "").startswith("application/json")
                else None
            ),
        )
    except Exception:
        return question, 0, None


def main():
    parser = argparse.ArgumentParser(
        description="Generate N questions and post to Modal endpoint (LLM-only)"
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.environ.get("MODAL_ENDPOINT_URL", ""),
        help="Endpoint URL (e.g., https://...modal.run)",
    )
    parser.add_argument(
        "--modal-key",
        default=os.environ.get("MODAL_PROXY_TOKEN_ID", ""),
        help="Modal Proxy Auth Token ID (sent as 'Modal-Key' header)",
    )
    parser.add_argument(
        "--modal-secret",
        default=os.environ.get("MODAL_PROXY_TOKEN_SECRET", ""),
        help="Modal Proxy Auth Token Secret (sent as 'Modal-Secret' header)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "8")),
        help="Concurrent requests",
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

    if not args.endpoint_url or not args.modal_key or not args.modal_secret:
        print(
            "Missing --endpoint-url or --modal-key or --modal-secret (or corresponding env vars)",
            file=sys.stderr,
        )
        sys.exit(2)
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY in environment", file=sys.stderr)
        sys.exit(2)

    questions = generate_questions(random.randint(1, 100), args.seed)

    successes = 0
    failures = 0

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [
                executor.submit(
                    post_question,
                    session,
                    args.endpoint_url,
                    args.modal_key,
                    args.modal_secret,
                    q,
                )
                for q in questions
            ]
            for fut in as_completed(futures):
                _, status_code, _ = fut.result()
                ok = 200 <= status_code < 300
                if ok:
                    successes += 1
                else:
                    failures += 1
    print(f"Completed. successes={successes} failures={failures}")
    if args.fail_on_error and failures > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

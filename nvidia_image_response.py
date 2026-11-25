import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time

from dotenv import load_dotenv
import requests


INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "mistralai/mistral-large-4-1t-a52b"
load_dotenv(Path(__file__).with_name(".env"))

TOPICS = [
    "bad tech support questions",
    "obvious life advice",
    "gym and dieting excuses",
    "corporate buzzwords and meetings",
    "dating app clichés",
    "terrible cooking attempts",
    "procrastination and deadlines",
    "bad drivers and traffic",
    "smartphones and social media addiction",
    "weather small talk",
    "broken sleep schedules",
    "impulse online shopping",
    "crypto and get-rich-quick schemes",
    "awkward social interactions",
    "airline travel nightmares",
    "coding bugs and git merge conflicts",
    "overpriced coffee shops",
    "group project slackers",
    "unsolicited life tips",
    "horoscopes and astrology",
    "roommate habits",
    "customer service complaints",
    "fitness influencers",
    "Monday mornings",
    "subscription services",
    "smart home devices failing",
    "job interview clichés",
    "binge-watching TV",
    "trying to save money",
    "basic common sense fails",
]

STYLES = [
    "dry, deadpan, minimal words",
    "overly dramatic and theatrical",
    "fake-cheerful customer service energy",
    "exhausted senior software engineer",
    "condescending Victorian aristocrat",
    "blunt, witty stand-up comedian",
    "passive-aggressive coworker",
    "unimpressed teenager",
    "nihilistic philosopher",
    "sharp, punchy one-liners",
]


def parse_json_array(content: str) -> list[dict]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"\A```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\Z", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except ValueError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[", cleaned):
        try:
            candidate, _ = decoder.raw_decode(cleaned[match.start() :])
        except ValueError:
            continue
        if isinstance(candidate, list):
            return candidate

    raise RuntimeError(
        f"Model did not return a JSON array. Preview: {cleaned[:200]!r}"
    )


def fetch_batch(
    api_key: str, batch_size: int, topic: str, style: str, max_retries: int = 3
) -> list[dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    prompt = (
        f"Generate exactly {batch_size} original, distinct question-and-answer pairs "
        f"for a sarcastic conversational assistant focused on '{topic}' with the tone: '{style}'. "
        "Return ONLY a valid JSON array, with each item having exactly the string fields "
        '"question" and "answer". Use natural, witty, context-aware replies with varied '
        "sentence structures. Avoid repeated punchlines, generic openers, hateful or "
        "discriminatory content, threats, and abusive insults. Do not include markdown or "
        "any text outside the JSON array."
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "stream": False,
        "temperature": 0.9,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
        "reasoning_effort": "low",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                INVOKE_URL, headers=headers, json=payload, timeout=(20, 120)
            )
            if response.status_code == 429:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            raw_pairs = parse_json_array(answer)

            return [
                {
                    "question": p["question"].strip(),
                    "answer": p["answer"].strip(),
                }
                for p in raw_pairs
                if isinstance(p, dict)
                and set(p) == {"question", "answer"}
                and isinstance(p["question"], str)
                and isinstance(p["answer"], str)
                and p["question"].strip()
                and p["answer"].strip()
            ]
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Batch failed after {max_retries} tries: {exc}"
                ) from exc
            time.sleep(1.5 * attempt)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sarcastic Q&A pairs via NVIDIA NIM."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6000,
        help="Total Q&A pairs to generate (default: 6000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Pairs per API call (default: 20).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent workers (default: 8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sarcastic_qa.jsonl"),
        help="Output JSONL file.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip() or getpass.getpass(
        "NVIDIA API key: "
    ).strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required.")

    seeds = list(product(TOPICS, STYLES))
    random.shuffle(seeds)

    num_batches = math.ceil((args.count * 1.1) / args.batch_size)
    tasks = [
        (args.batch_size, *seeds[i % len(seeds)]) for i in range(num_batches)
    ]

    seen_questions: set[str] = set()
    total_saved = 0
    lock = threading.Lock()
    args.output.write_text("", encoding="utf-8")

    print(
        f"Generating {args.count} pairs across {num_batches} batches ({args.workers} workers)..."
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(fetch_batch, api_key, size, topic, style)
            for size, topic, style in tasks
        ]

        for future in as_completed(futures):
            if total_saved >= args.count:
                continue
            try:
                batch = future.result()
            except Exception as exc:
                print(f"[Warning] {exc}")
                continue

            with lock:
                new_items = []
                for pair in batch:
                    if total_saved + len(new_items) >= args.count:
                        break
                    norm = re.sub(
                        r"\W+", " ", pair["question"].casefold()
                    ).strip()
                    if norm and norm not in seen_questions:
                        seen_questions.add(norm)
                        new_items.append(pair)

                if new_items:
                    with args.output.open("a", encoding="utf-8") as f:
                        for item in new_items:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    total_saved += len(new_items)
                    print(f"Saved: {total_saved}/{args.count}")

    print(f"Done! {total_saved} unique pairs written to {args.output}")


if __name__ == "__main__":
    main()

"""send_to_chatgpt.py

Send prompts to the OpenAI API and collect "stay"/"swerve" decisions.

Design goals:
- Input: list of 13 prompt strings
- Output: list of 13 strings, each exactly "stay" or "swerve"
- Uses Structured Outputs (JSON schema) to force valid output.
"""



# send_to_chatgpt.py
import json
import time
import random
import openai
from openai import OpenAI

SYSTEM_INSTRUCTIONS = (
    "You will be given a list of 13 scenarios.\n"
    "For each scenario, decide which action is morally preferable: stay or swerve.\n"
    "Return ONLY valid JSON matching the provided schema.\n"
)

BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "enum": ["stay", "swerve"]},
        }
    },
    "required": ["choices"],
    "additionalProperties": False,
}

def _retry_after_seconds(e: Exception) -> float | None:
    """
    Try to respect Retry-After headers if present.
    """
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None

    # Some APIs use retry-after-ms; OpenAI commonly uses retry-after
    ra_ms = headers.get("retry-after-ms")
    if ra_ms:
        try:
            return float(ra_ms) / 1000.0
        except Exception:
            pass

    ra = headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except Exception:
            pass

    return None


def _make_user_prompt(prompts: list[str]) -> str:
    # Keep it compact to reduce token usage
    lines = []
    for i, p in enumerate(prompts, start=1):
        lines.append(f"{i}. {p}")
    return (
        "For each item, decide 'stay' or 'swerve'. "
        "Return an array of choices in the same order.\n\n"
        + "\n\n".join(lines)
    )


def judge_prompts_batch(
    prompts: list[str],
    model: str = "gpt-4o-mini",
    max_attempts: int = 8,
) -> list[str]:
    """
    Single batch call. Returns list of 'stay'/'swerve' in same order as prompts.
    Includes robust handling for 429 rate limits.
    """

    client = OpenAI(max_retries=0)  # IMPORTANT: we control retries here

    user_prompt = _make_user_prompt(prompts)

    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "batch_decisions",
                        "strict": True,
                        "schema": BATCH_SCHEMA,
                    }
                },
                temperature=0,
                max_output_tokens=80,
                timeout=60,
            )

            data = json.loads(resp.output_text)
            choices = [c.strip().lower() for c in data["choices"]]

            # Validate
            for c in choices:
                if c not in ("stay", "swerve"):
                    raise ValueError(f"Bad choice from model: {c!r}")

            if len(choices) != len(prompts):
                raise ValueError(
                    f"Returned {len(choices)} choices for {len(prompts)} prompts."
                )

            return choices

        except openai.RateLimitError as e:
            # This prints the *real* reason (rate limit vs insufficient quota)
            print("\n🚨 RateLimitError (429)")
            if hasattr(e, "body") and e.body:
                print("Error body:", e.body)

            # If it's insufficient quota, retrying won't help
            if hasattr(e, "body") and e.body:
                err = e.body.get("error", {}) if isinstance(e.body, dict) else {}
                if err.get("code") == "insufficient_quota":
                    raise RuntimeError(
                        "Your API quota is exceeded / no credits left. "
                        "Go to OpenAI dashboard → Billing/Usage and add credits or raise limits."
                    ) from e

            wait_s = _retry_after_seconds(e)
            if wait_s is None:
                # exponential backoff with jitter
                wait_s = min(60, (2 ** (attempt - 1))) + random.random()

            print(f"⏳ Waiting {wait_s:.2f}s then retrying... (attempt {attempt}/{max_attempts})")
            time.sleep(wait_s)

        except openai.APIStatusError as e:
            # other transient errors
            print("\n🚨 APIStatusError")
            print("Status:", getattr(e, "status_code", None))
            if hasattr(e, "body") and e.body:
                print("Error body:", e.body)
            raise

    raise RuntimeError("Failed after max attempts due to repeated rate limits.")


import json
import time
from groq import Groq, RateLimitError
from pydantic import ValidationError
from schemas import ExtractedProjectSpec

MODEL = "qwen/qwen3.6-27b"
MAX_OUTPUT_TOKENS = 500

SYSTEM_PROMPT = """You are an expert software project analyst. Read the client's
project request and extract a complete, structured project specification.

Only include information explicitly stated or clearly implied in the text -- do not
invent requirements, technologies, or numbers that are not present. Use empty lists
[] for fields with no information, and null for deadline_days if no deadline is
mentioned. Be concise -- short phrases, not full sentences, in each list item.

Respond with ONLY a valid JSON object, no other text, matching EXACTLY this structure:
{
  "project_name": "<string>", "domain": "<string>",
  "application_type": ["<string>"], "functional_requirements": ["<string>"],
  "technologies": ["<string>"], "integrations": ["<string>"], "database": ["<string>"],
  "security_requirements": ["<string>"], "ai_ml_requirements": ["<string>"],
  "cloud_requirements": ["<string>"], "data_requirements": ["<string>"],
  "real_time_requirements": ["<string>"], "deadline_days": null, "constraints": ["<string>"]
}"""

def extract_features(raw_text: str, max_retries: int = 2) -> ExtractedProjectSpec:
    client = Groq()
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort="none",
            )
        except RateLimitError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(20)
                continue
            raise RuntimeError(f"Groq rate limit hit after {max_retries + 1} attempts: {e}")

        raw_content = completion.choices[0].message.content
        try:
            data = json.loads(raw_content)
            return ExtractedProjectSpec(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            continue
    raise RuntimeError(f"Extraction failed after {max_retries + 1} attempts: {last_error}")

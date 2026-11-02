"""LLM client + JSON helpers.

Configurable, OpenAI-compatible. Defaults to Groq's free LLaMA 3.3 70B
(no paid OpenAI key needed). Also works with OpenAI, Together, Ollama, etc.
"""
import json
import re

from openai import OpenAI

_client = None
_model = None


def _init():
    global _client, _model
    api_key = os.getenv("LLM_API_KEY")  # noqa: F821 (os imported below)
    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY not set. Copy .env.example to .env and add a free Groq key "
            "from https://console.groq.com/keys"
        )
    _client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),  # noqa
    )
    _model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")  # noqa


# lazy import of os kept at bottom for readability; but we need it at runtime:
import os  # noqa: E402


def chat(system: str, user: str, temperature: float = 0.3, max_tokens: int = 1200) -> str:
    """Plain text completion."""
    if _client is None:
        _init()
    resp = _client.chat.completions.create(
        model=_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _extract_json(text: str):
    """Pull the first balanced JSON object/array out of an LLM response."""
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        raise ValueError("no JSON found in response")
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in response")


def chat_json(system: str, user: str, temperature: float = 0.2, max_tokens: int = 1600):
    """Completion that is parsed as JSON. Falls back to None on parse failure."""
    raw = chat(system, user, temperature=temperature, max_tokens=max_tokens)
    try:
        return _extract_json(raw)
    except Exception:
        return None

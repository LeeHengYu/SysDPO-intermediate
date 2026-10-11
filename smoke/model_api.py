"""Small, stateless OpenRouter interface. One invocation means one HTTP request."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def call_model(model, messages, *, temperature=0.3, max_tokens=8192, json_output=False):
    key = os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_KEY is not available in this process")

    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    if json_output:
        payload["response_format"] = {"type": "json_object"}
        payload["reasoning"] = {"enabled": False}

    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )

    # No retry: keep each judge invocation single-shot and failures visible.
    try:
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(f"OpenRouter HTTP {error.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("OpenRouter connection failed or timed out") from None

    if not isinstance(result, dict):
        raise RuntimeError("OpenRouter response is not an object")
    return result  # Caller saves raw output/usage before validating the completion.


def response_text(result, *, allow_truncated=False):
    if "error" in result:
        raise ValueError("OpenRouter returned an API error")

    try:
        choice = result["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("Missing completion text") from None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty completion text")

    allowed_reasons = ("stop", "length") if allow_truncated else ("stop",)
    if choice.get("finish_reason") not in allowed_reasons:
        raise ValueError(f"Incomplete completion: {choice.get('finish_reason')}")

    usage = result.get("usage", {})
    if any(type(usage.get(name)) is not int or usage[name] < 0
           for name in ("prompt_tokens", "completion_tokens", "total_tokens")):
        raise ValueError("Missing or invalid token usage")

    return text

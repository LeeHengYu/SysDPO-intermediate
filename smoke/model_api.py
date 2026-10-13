"""Small, stateless OpenRouter interface. One invocation means one HTTP request."""

import json
import os
import httpx
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError, ResponseValidationError


def call_model(model, messages, *, temperature=0.3, max_tokens=8192, json_output=False):
    key = os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_KEY is not available in this process")

    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    if json_output:
        payload["response_format"] = {"type": "json_object"}
        payload["reasoning"] = {"enabled": False}

    if model == "microsoft/microsoft-decision-1":
        payload = {"model": model, "state": json.loads(messages[1]["content"]),
                   "questions": {"preference": {
                       "type": "choice", "instructions": messages[0]["content"],
                       "criteria": {
                           "option1": "candidate_1 is preferable in correctness, completeness, adherence to the task's required output format, or downstream usefulness; at equal quality it is meaningfully more concise.",
                           "option2": "candidate_2 is preferable in correctness, completeness, adherence to the task's required output format, or downstream usefulness; at equal quality it is meaningfully more concise.",
                           "tie": "Neither candidate has a clear advantage in correctness, completeness, required output format, downstream usefulness, or conciseness, or the preference cannot be assessed reliably.",
                       }}}}

    # Capture the raw JSON so SDK parsing cannot drop provider or usage metadata.
    responses = []
    try:
        with httpx.Client(event_hooks={"response": [responses.append]}) as http_client:
            with OpenRouter(api_key=key, client=http_client, retry_config=None,
                            timeout_ms=600_000) as client:
                if model == "microsoft/microsoft-decision-1":
                    client.alpha.decisions.create(**payload)
                else:
                    client.chat.send(**payload)
                result = responses[-1].json()
    except ResponseValidationError as error:
        result = error.raw_response.json()
    except OpenRouterError as error:
        raise RuntimeError(f"OpenRouter HTTP {error.status_code}") from None
    except httpx.RequestError:
        raise RuntimeError("OpenRouter connection failed or timed out") from None

    if not isinstance(result, dict):
        raise RuntimeError("OpenRouter response is not an object")
    return result  # Caller saves raw output/usage before validating the completion.


def response_text(result, *, allow_truncated=False):
    if "error" in result:
        raise ValueError("OpenRouter returned an API error")

    if "answers" in result:
        answer = result["answers"]["preference"]
        if answer.get("type") != "choice" or answer.get("choice") not in ("option1", "option2", "tie"):
            raise ValueError("Invalid Decisions preference answer")
        usage = result.get("usage", {})
        if any(type(usage.get(name)) is not int or usage[name] < 0
               for name in ("input_tokens", "output_tokens")):
            raise ValueError("Missing or invalid Decisions token usage")
        return json.dumps({"choice": answer["choice"]})

    choice = result["choices"][0]
    text = choice["message"]["content"]
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

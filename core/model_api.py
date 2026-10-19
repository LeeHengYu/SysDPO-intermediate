"""Small, stateless OpenRouter interface. One invocation means one HTTP request."""

import json
import os
import httpx
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError, ResponseValidationError


def call_judge(state, rubric, *, model="microsoft/microsoft-decision-1"):
    """Judge an intermediate pair independently of how A generated it.

    state contains candidate_1, candidate_2 and their task context.
    Returns the raw Decisions response, including choice and usage.
    """
    questions = {"preference": {
        "type": "choice", "instructions": rubric,
        "criteria": {
            "option1": "candidate_1 is preferable in correctness, completeness, adherence to the task's required output format, or downstream usefulness; at equal quality it is meaningfully more concise.",
            "option2": "candidate_2 is preferable in correctness, completeness, adherence to the task's required output format, or downstream usefulness; at equal quality it is meaningfully more concise.",
        }}}
    return request_model(lambda client: client.alpha.decisions.create(
        model=model, state=state, questions=questions))


def request_model(send):
    """Shared SDK transport; one invocation makes one HTTP attempt."""
    key = os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_KEY is not available in this process")

    # Capture the raw JSON so SDK parsing cannot drop provider or usage metadata.
    responses = []
    try:
        with httpx.Client(event_hooks={"response": [responses.append]}) as http_client:
            with OpenRouter(api_key=key, client=http_client, retry_config=None,
                            timeout_ms=600_000) as client:
                send(client)
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


def response_text(result):
    if "error" in result:
        raise ValueError("OpenRouter returned an API error")

    answer = result["answers"]["preference"]
    if answer.get("type") != "choice" or answer.get("choice") not in ("option1", "option2"):
        raise ValueError("Invalid Decisions preference answer")
    usage = result.get("usage", {})
    if any(type(usage.get(name)) is not int or usage[name] < 0
           for name in ("input_tokens", "output_tokens")):
        raise ValueError("Missing or invalid Decisions token usage")

    return json.dumps({"choice": answer["choice"]})

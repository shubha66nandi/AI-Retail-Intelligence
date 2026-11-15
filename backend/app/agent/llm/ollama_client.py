
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:0.6b"


def generate_response(
    user_prompt: str,
    system_prompt: str = "You are an AI Retail Data Analyst."
) -> str:
    """
    Generate a response using the local Qwen model through Ollama.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False
    }

    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result["message"]["content"].strip()

    except HTTPError as error:
        raise RuntimeError(
            f"Ollama API error: {error.code} - {error.reason}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        ) from error

    except TimeoutError as error:
        raise RuntimeError("Ollama request timed out.") from error

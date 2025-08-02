from __future__ import annotations

"""Utility for translating text into Traditional Chinese (Taiwan)."""

from typing import Dict

from openai import OpenAI

ZH_SYSTEM_PROMPT = (
    "You are a professional financial translator. "
    "Translate the user's text into Traditional Chinese used in Taiwan. "
    "Preserve numbers, symbols, and code snippets. Only return the translated text."
)


def translate_to_zh(text: str, api_config: Dict) -> str:
    """Translate English ``text`` to Traditional Chinese using an LLM.

    Parameters
    ----------
    text: str
        The text to translate.
    api_config: Dict
        Configuration containing at least ``backend_url``, ``api_key`` and
        ``quick_think_llm`` for the model.

    Returns
    -------
    str
        The translated text. If translation fails, returns the original ``text``.
    """
    if not text:
        return ""

    try:
        client = OpenAI(
            base_url=api_config.get("backend_url"), api_key=api_config.get("api_key")
        )

        response = client.responses.create(
            model=api_config.get("quick_think_llm", "gpt-4o-mini"),
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": ZH_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            ],
            max_output_tokens=800,
        )

        # The first output element contains the model's text response
        return response.output[0].content[0].text
    except Exception as e:  # pragma: no cover - network/LLM failures handled gracefully
        print(f"[WARNING] Translation failed: {e}")
        return text

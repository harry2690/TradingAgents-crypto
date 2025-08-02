from __future__ import annotations

"""Utility for translating text into Traditional Chinese (Taiwan)."""

from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

ZH_SYSTEM_PROMPT = (
    "You are a professional financial translator. "
    "Translate the user's text into Traditional Chinese used in Taiwan. "
    "Preserve numbers, symbols, and code snippets. Only return the translated text."
)


def _create_llm_client(api_config: Dict):
    """Create an LLM client based on the provider in ``api_config``."""
    provider = (api_config.get("llm_provider") or "openai").lower()
    model = api_config.get("quick_think_llm", "gpt-4o-mini")

    if provider in {"openai", "ollama", "openrouter"}:
        return ChatOpenAI(
            model=model,
            base_url=api_config.get("backend_url"),
            api_key=api_config.get("api_key"),
        )
    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            base_url=api_config.get("backend_url"),
            api_key=api_config.get("api_key"),
        )
    if provider == "google":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_config.get("api_key"),
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _invoke_client(client, text: str) -> str:
    """Send translation prompt to the LLM client and return response text."""
    messages = [
        SystemMessage(content=ZH_SYSTEM_PROMPT),
        HumanMessage(content=text),
    ]

    response = client.invoke(messages)
    content = getattr(response, "content", response)

    if isinstance(content, list):
        # Some providers may return a list of parts
        return "".join(str(part) for part in content)
    return str(content)


def translate_to_zh(text: str, api_config: Dict) -> str:
    """Translate English ``text`` to Traditional Chinese using an LLM.

    Parameters
    ----------
    text: str
        The text to translate.
    api_config: Dict
        Configuration containing at least ``backend_url``, ``api_key``,
        ``quick_think_llm`` for the model, and ``llm_provider``.

    Returns
    -------
    str
        The translated text. If translation fails, returns the original ``text``.
    """
    if not text:
        return ""

    try:
        client = _create_llm_client(api_config)
        return _invoke_client(client, text)
    except Exception as e:  # pragma: no cover - network/LLM failures handled gracefully
        print(f"[WARNING] Translation failed: {e}")
        return text

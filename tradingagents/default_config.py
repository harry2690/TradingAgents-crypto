import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": os.getenv("TRADINGAGENTS_DATA_DIR", "./data"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "anthropic",
    "deep_think_llm": "claude-3-sonnet-20240229",
    "quick_think_llm": "claude-3-haiku-20240307",
    "backend_url": "https://api.anthropic.com",
    "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
    "embedding_backend_url": os.getenv(
        "EMBEDDING_BACKEND_URL", "https://api.openai.com/v1"
    ),
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
    "finnhub_api_key": os.getenv("FINNHUB_API_KEY", ""),
}

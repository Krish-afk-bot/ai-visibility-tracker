import os
import time
import random
import logging
from threading import Lock
from typing import Dict, List, Optional, Tuple, Any

from openai import OpenAI, APIStatusError, RateLimitError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)

# Base URLs for OpenAI-compatible providers
BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/"
}


class LLMClientError(Exception):
    """Base exception for LLM client failures."""
    pass


class ConfigurationError(LLMClientError):
    """Raised when an API key or configuration is missing or invalid."""
    pass


class ModelUnavailableError(LLMClientError):
    """Raised when a requested model is retired, not found, or unavailable."""
    pass


class RateLimitExceededError(LLMClientError):
    """Raised when rate limit is exceeded and retries are exhausted."""
    pass


class ProviderRateLimiter:
    """Thread-safe Token Bucket / Window rate limiter for a specific provider."""

    def __init__(self, requests_per_minute: int = 20):
        self.rpm = max(1, requests_per_minute)
        self.lock = Lock()
        self.timestamps: List[float] = []

    def acquire(self):
        """Block until a request slot is available in the current 60-second window."""
        with self.lock:
            now = time.time()
            self.timestamps = [t for t in self.timestamps if now - t < 60.0]
            if len(self.timestamps) >= self.rpm:
                oldest = self.timestamps[0]
                sleep_needed = max(0.05, 60.0 - (now - oldest) + 0.05)
                logger.info(f"Rate limit reached ({self.rpm} RPM). Sleeping for {sleep_needed:.2f}s")
                time.sleep(sleep_needed)
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < 60.0]

            self.timestamps.append(time.time())


class LLMClientManager:
    """Singleton-style manager holding cached OpenAI client instances and rate limiters."""

    _instance: Optional["LLMClientManager"] = None
    _init_lock = Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(LLMClientManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._clients: Dict[str, OpenAI] = {}
        self._client_keys: Dict[str, str] = {}
        self._rate_limiters: Dict[str, ProviderRateLimiter] = {}
        self._client_lock = Lock()
        self._initialized = True

    def get_rate_limiter(self, provider: str) -> ProviderRateLimiter:
        provider = provider.lower()
        with self._client_lock:
            if provider not in self._rate_limiters:
                env_rpm = os.getenv(f"{provider.upper()}_RPM")
                if provider == "openrouter":
                    default_rpm = 20
                elif provider == "gemini":
                    default_rpm = 15  # Google AI Studio free tier: 15 RPM
                else:
                    default_rpm = 30
                try:
                    rpm = int(env_rpm) if env_rpm else default_rpm
                except ValueError:
                    rpm = default_rpm
                self._rate_limiters[provider] = ProviderRateLimiter(rpm)
            return self._rate_limiters[provider]

    def get_client(self, provider: str) -> OpenAI:
        """Return or create a cached OpenAI client for the given provider."""
        provider = provider.lower()
        if provider not in BASE_URLS:
            raise ConfigurationError(f"Unsupported provider: '{provider}'. Supported: {list(BASE_URLS.keys())}")

        api_key = os.getenv(f"{provider.upper()}_API_KEY")
        if not api_key or not api_key.strip():
            raise ConfigurationError(
                f"Missing API key for provider '{provider}'. Please set {provider.upper()}_API_KEY in your environment or .env file."
            )
        api_key = api_key.strip()

        with self._client_lock:
            cached = self._clients.get(provider)
            # Reuse the client only while it is backed by the current key, so a
            # key supplied while the app is running takes effect on the next call.
            if cached is not None and self._client_keys.get(provider) == api_key:
                return cached

            default_headers: Dict[str, str] = {}
            if provider == "openrouter":
                referer = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8501")
                title = os.getenv("OPENROUTER_TITLE", "AI Visibility Tracker")
                if referer:
                    default_headers["HTTP-Referer"] = referer
                if title:
                    default_headers["X-Title"] = title

            timeout_val = float(os.getenv(f"{provider.upper()}_TIMEOUT", "45.0"))
            client = OpenAI(
                base_url=BASE_URLS[provider],
                api_key=api_key,
                default_headers=default_headers if default_headers else None,
                timeout=timeout_val,
                max_retries=0
            )
            self._clients[provider] = client
            self._client_keys[provider] = api_key
            return client

    def reset_clients(self):
        """Reset cached clients (useful for testing or config changes)."""
        with self._client_lock:
            self._clients.clear()
            self._client_keys.clear()


# Global client manager instance
client_manager = LLMClientManager()


def parse_provider_model(provider_model_str: str) -> Tuple[str, str]:
    """Parse 'provider:model' string into (provider, model)."""
    if not provider_model_str or ":" not in provider_model_str:
        raise ValueError(f"Invalid format '{provider_model_str}'. Expected 'provider:model', e.g. 'groq:llama-3.1-8b-instant'")
    parts = provider_model_str.split(":", 1)
    return parts[0].strip().lower(), parts[1].strip()


def get_available_models() -> List[str]:
    """Return list of available 'provider:model' strings based on .env config.
    
    A provider is considered available only if its key and at least one model are set.
    """
    available = []
    
    # OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    or_models_str = os.getenv("OPENROUTER_MODELS", "").strip()
    if or_key and or_models_str:
        for m in [m.strip() for m in or_models_str.split(",") if m.strip()]:
            available.append(f"openrouter:{m}")

    # Groq
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_models_str = os.getenv("GROQ_MODELS", "").strip()
    if groq_key and groq_models_str:
        for m in [m.strip() for m in groq_models_str.split(",") if m.strip()]:
            available.append(f"groq:{m}")

    # Gemini (via official OpenAI-compatible endpoint, only when both key and models are set)
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_models_str = os.getenv("GEMINI_MODELS", "").strip()
    if gemini_key and gemini_models_str:
        for m in [m.strip() for m in gemini_models_str.split(",") if m.strip()]:
            available.append(f"gemini:{m}")

    return available


def get_analyzer_model() -> Optional[str]:
    """Return the configured analyzer model (provider:model), or None if unset.
    
    ANALYZER_MODEL is strictly required and does NOT fall back to a tested model.
    """
    configured = os.getenv("ANALYZER_MODEL", "").strip()
    return configured if configured else None


def validate_analyzer_config(analyzer_model: Optional[str], selected_models: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """Validate analyzer model configuration.
    
    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    if not analyzer_model or not analyzer_model.strip():
        return False, "ANALYZER_MODEL is required. Please configure it in your .env (e.g. 'groq:<model-id>' or 'gemini:<model-id>')."

    if ":" not in analyzer_model:
        return False, f"Invalid ANALYZER_MODEL '{analyzer_model}'. Format must be 'provider:model-id' (e.g. 'groq:llama-3.1-8b-instant')."

    provider, model = parse_provider_model(analyzer_model)
    if provider not in BASE_URLS:
        return False, f"ANALYZER_MODEL uses unsupported provider '{provider}'. Supported: {list(BASE_URLS.keys())}."

    api_key = os.getenv(f"{provider.upper()}_API_KEY", "").strip()
    if not api_key:
        return False, f"Missing API key for analyzer provider '{provider}'. Please set {provider.upper()}_API_KEY in .env."

    if selected_models and analyzer_model in selected_models:
        return False, f"ANALYZER_MODEL ('{analyzer_model}') cannot be in the list of selected test models. A model must never grade its own answers."

    return True, None


def query_ai(
    prompt: str,
    provider: str,
    model: str,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    temperature: float = 0.3
) -> str:
    """Execute raw prompt against provider and model without system wrappers or brand bias.
    
    Raises:
        ConfigurationError: if API key or provider is missing
        ModelUnavailableError: if model is retired/not found (404)
        RateLimitExceededError: if 429 persists after retries
        LLMClientError: on other failures
    """
    if not prompt or not prompt.strip():
        raise LLMClientError("Prompt cannot be empty.")

    provider = provider.lower()
    client = client_manager.get_client(provider)
    rate_limiter = client_manager.get_rate_limiter(provider)

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            rate_limiter.acquire()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )

            if not response.choices or not response.choices[0].message:
                raise LLMClientError(f"Empty response from {provider}:{model}")

            content = response.choices[0].message.content
            if content is None or not content.strip():
                raise LLMClientError(f"Model {provider}:{model} returned an empty completion.")

            logger.info(f"{provider}:{model} responded ({len(content)} chars)")
            return content.strip()

        except RateLimitError as e:
            last_error = e
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                ra_header = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
                if ra_header:
                    try:
                        retry_after = float(ra_header)
                    except ValueError:
                        pass

            if attempt < max_retries:
                sleep_time = retry_after if retry_after is not None else (initial_backoff * (2 ** attempt)) + random.uniform(0.1, 0.6)
                logger.warning(
                    f"Rate limit 429 on {provider}:{model} (attempt {attempt + 1}/{max_retries + 1}). Sleeping {sleep_time:.2f}s."
                )
                time.sleep(sleep_time)
                continue
            else:
                raise RateLimitExceededError(
                    f"Rate limit exceeded on {provider}:{model} after {max_retries} retries: {str(e)}"
                ) from e

        except APIStatusError as e:
            status_code = getattr(e, "status_code", None)
            if status_code == 404:
                raise ModelUnavailableError(
                    f"Model '{model}' is unavailable, not found, or retired on {provider} (HTTP 404): {e.message}"
                ) from e
            elif status_code is not None and 500 <= status_code < 600:
                last_error = e
                if attempt < max_retries:
                    sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(0.1, 0.6)
                    logger.warning(
                        f"Server error {status_code} on {provider}:{model}. Retrying in {sleep_time:.2f}s."
                    )
                    time.sleep(sleep_time)
                    continue
                else:
                    raise LLMClientError(
                        f"Server error {status_code} from {provider}:{model} after {max_retries} retries: {str(e)}"
                    ) from e
            else:
                raise LLMClientError(f"API status error ({status_code}) on {provider}:{model}: {str(e)}") from e

        except (APITimeoutError, APIConnectionError) as e:
            last_error = e
            if attempt < max_retries:
                sleep_time = (initial_backoff * (2 ** attempt)) + random.uniform(0.1, 0.6)
                logger.warning(f"Network/timeout error on {provider}:{model}. Retrying in {sleep_time:.2f}s.")
                time.sleep(sleep_time)
                continue
            else:
                raise LLMClientError(f"Network or timeout failure on {provider}:{model}: {str(e)}") from e

        except Exception as e:
            if isinstance(e, LLMClientError):
                raise
            raise LLMClientError(f"Unexpected error communicating with {provider}:{model}: {str(e)}") from e

    raise LLMClientError(f"Failed to query {provider}:{model}: {str(last_error)}")

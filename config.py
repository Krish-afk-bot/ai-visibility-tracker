import os


def get_setting(key: str, default: str = "") -> str:
    """Read a setting from Streamlit secrets or environment variables.

    Priority: ST.secrets > os.environ. This allows the same code to work
    locally (via .env or Streamlit secrets) and on Streamlit Cloud (via
    App Settings → Secrets) without changes.
    """
    # 1. Try Streamlit secrets first (Cloud / test-injected)
    try:
        import streamlit as st  # available at runtime; guard for import-time

        secrets = getattr(st, "secrets", None)
        if secrets is not None:
            val = secrets.get(key)
            # Only trust plain scalar values; a missing secrets file can
            # surface as an empty mapping or an exception.
            if isinstance(val, (str, int, float, bool)) and str(val).strip():
                return str(val)
    except Exception:
        pass

    # 2. Fall back to environment variable
    val = os.getenv(key, default)
    return str(val)


def get_int_setting(key: str, default: int = 0) -> int:
    """Read an integer setting from secrets or environment."""
    val = get_setting(key, str(default))
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def get_float_setting(key: str, default: float = 0.0) -> float:
    """Read a float setting from secrets or environment."""
    val = get_setting(key, str(default))
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
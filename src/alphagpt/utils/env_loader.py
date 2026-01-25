"""Environment variable loader utility"""

import os
from typing import Optional


def get_env_or_config(env_key: str, config_value: Optional[str]) -> Optional[str]:
    """
    Get value from environment variable or config file

    Priority: Environment variable > Config value

    Args:
        env_key: Environment variable name
        config_value: Value from config file

    Returns:
        Value from environment variable if exists, otherwise config value

    Example:
        >>> api_key = get_env_or_config('DEEPSEEK_API_KEY', config.api_key)
    """
    env_value = os.getenv(env_key)
    if env_value:
        return env_value
    return config_value

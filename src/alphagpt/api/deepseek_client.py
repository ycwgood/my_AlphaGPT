"""DeepSeek API client for factor generation"""

import logging
import time
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Client for interacting with DeepSeek API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        use_api: bool = True
    ):
        """
        Initialize DeepSeek API client

        Args:
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            model: Model name to use
            use_api: Whether to use API or fallback to local generation
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.use_api = use_api

        # Validate API key if API is enabled
        if self.use_api and not self.api_key:
            raise ValueError(
                "DeepSeek API key is required when use_api=True. "
                "Set via config.yaml or DEEPSEEK_API_KEY environment variable"
            )

        # Initialize OpenAI client for DeepSeek
        if self.use_api:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def call_api(
        self,
        prompt: str,
        max_retries: int = 3,
        timeout: int = 120,
        temperature: float = 0.8,
        max_tokens: int = 8000
    ) -> Optional[str]:
        """
        Call DeepSeek API with retry logic using OpenAI SDK

        Args:
            prompt: Input prompt for the model
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            API response content or None if failed
        """
        if not self.use_api:
            logger.info("API调用被禁用，使用本地生成器")
            return None

        for attempt in range(max_retries):
            try:
                logger.info(f"调用DeepSeek API (尝试 {attempt+1}/{max_retries})，超时: {timeout}秒...")

                # Call DeepSeek API using OpenAI SDK
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    timeout=timeout
                )

                # Extract content from response
                content = response.choices[0].message.content

                logger.info(f"DeepSeek API调用成功，响应长度: {len(content)} 字符")
                return content

            except Exception as e:
                logger.error(f"DeepSeek API调用异常 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue

        logger.error("所有API调用尝试均失败")
        return None

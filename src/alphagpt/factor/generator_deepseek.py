"""Factor generator module - Part 1: Core class and initialization"""

import logging
import random
import json
import re
import numpy as np
import numexpr as ne
from typing import List, Dict, Optional

from ..api import DeepSeekClient
from .operators import FactorOperators
from .calculator import FactorCalculator
from ..utils.exceptions import FactorGenerationError

logger = logging.getLogger(__name__)


class FactorGenerator:
    """Generate quantitative factors using DeepSeek API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        use_api: bool = True
    ):
        """
        Initialize factor generator

        Args:
            api_key: API key for DeepSeek
            base_url: API base URL
            model: Model name
            use_api: Whether to use API or local generation
        """
        self.api_client = DeepSeekClient(api_key, base_url, model, use_api)
        self.ops = FactorOperators.get_operators()
        self.generated_factors_cache: List[str] = []
        self._factor_values_cache: Dict[str, np.ndarray] = {}
        self.calculator = FactorCalculator()

    def is_similar_factor(
        self,
        new_factor: str,
        existing_factors: List[str],
        feature_data: Dict[str, np.ndarray],
        threshold: float = 0.9
    ) -> bool:
        """
        检查新因子是否与已有因子高度相似

        Args:
            new_factor: 新因子表达式
            existing_factors: 已有因子列表
            feature_data: 特征数据
            threshold: 相关性阈值

        Returns:
            True 如果相似，False 否则
        """
        if not existing_factors or not feature_data:
            return False

        # 计算新因子的值
        new_values = self.calculator.calculate_factor_value(new_factor, feature_data)
        if new_values is None:
            return True  # 无法计算的因子视为相似（应被过滤）

        for existing in existing_factors:
            # 使用缓存的因子值
            if existing in self._factor_values_cache:
                existing_values = self._factor_values_cache[existing]
            else:
                existing_values = self.calculator.calculate_factor_value(existing, feature_data)
                if existing_values is not None:
                    self._factor_values_cache[existing] = existing_values

            if existing_values is None:
                continue

            # 计算相关系数
            try:
                corr = np.corrcoef(new_values, existing_values)[0, 1]
                if not np.isnan(corr) and abs(corr) > threshold:
                    return True
            except Exception:
                continue

        return False

    def add_to_cache(self, factor: str) -> None:
        """添加因子到缓存"""
        if factor not in self.generated_factors_cache:
            self.generated_factors_cache.append(factor)

    def clear_cache(self) -> None:
        """清空因子缓存"""
        self.generated_factors_cache.clear()
        self._factor_values_cache.clear()

    def validate_factor_expression(self, expr: str, features: List[str]) -> bool:
        """
        Validate factor expression

        Args:
            expr: Factor expression
            features: List of available features

        Returns:
            True if valid, False otherwise
        """
        if not expr or len(expr) < 5 or len(expr) > 200:
            return False

        # Check if contains at least one feature
        has_feature = any(feat in expr for feat in features)
        if not has_feature:
            return False

        # Check bracket matching
        if expr.count('(') != expr.count(')'):
            return False

        # Check division by zero protection
        if '/0' in expr.replace(' ', ''):
            return False

        return True

    def generate_factors_local(self, features: List[str], num_factors: int = 10) -> List[str]:
        """
        Generate factors locally without API

        Args:
            features: List of feature names
            num_factors: Number of factors to generate

        Returns:
            List of factor expressions
        """
        logger.info("使用本地生成器创建因子...")

        # 预先过滤特征子集，避免空列表异常
        vol_ret_features = [f for f in features if 'vol' in f or 'ret' in f] or features
        trend_mom_features = [f for f in features if 'trend' in f or 'mom' in f] or features

        # Define local factor templates
        factor_templates = [
            # Momentum factors
            lambda: f"{random.choice(features)} + {random.choice(features)} * {random.choice(features)}",
            lambda: f"{random.choice(features)} - {random.choice(['NEG', 'ABS', 'DELTA'])}({random.choice(features)})",
            # Reversal factors
            lambda: f"NEG({random.choice(features)}) * {random.choice(features)}",
            lambda: f"{random.choice(features)} / ({random.choice(features)} + 1e-6)",
            # Volume-price factors (使用预过滤的特征子集)
            lambda: f"{random.choice(vol_ret_features)} * {random.choice(trend_mom_features)}",
            # Volatility factors
            lambda: f"ABS(DELTA({random.choice(features)})) + {random.choice(features)}",
            lambda: f"{random.choice(features)} - DELTA({random.choice(features)})",
            # Composite factors
            lambda: f"({random.choice(features)} * {random.choice(features)}) / "
                    f"({random.choice(features)} + 1e-6)",
        ]

        factors = []
        seen = set()

        for _ in range(num_factors * 2):
            if len(factors) >= num_factors:
                break

            template = random.choice(factor_templates)
            factor = template()

            if (self.validate_factor_expression(factor, features) and
                factor not in seen and
                len(factor) < 100):
                factors.append(factor)
                seen.add(factor)

        logger.info(f"本地生成器创建了 {len(factors)} 个因子")
        return factors[:num_factors]

    def parse_factor_expressions(self, api_response: str, features: List[str]) -> Dict:
        """
        Parse factor expressions from API response

        Args:
            api_response: Raw API response
            features: List of available features

        Returns:
            Dictionary with generated_factors key
        """
        try:
            # Try to parse as JSON
            try:
                return json.loads(api_response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                json_match = re.search(r'```json\n({.*?})\n```', api_response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Extract plain text factor expressions
                factor_exprs = []
                expr_pattern = r'[A-Za-z0-9_]+(\s*[\+\-\*\/]\s*[A-Za-z0-9_]+|\s*\([^)]+\))+'
                matches = re.findall(expr_pattern, api_response)
                for match in matches:
                    clean_expr = match.strip()
                    if clean_expr and any(f in clean_expr for f in features):
                        factor_exprs.append(clean_expr)

                return {"generated_factors": factor_exprs}

        except Exception as e:
            logger.error(f"解析因子表达式失败: {str(e)}")
            return {"generated_factors": []}

    def _build_prompt(
        self,
        features: List[str],
        max_seq_len: int,
        min_op_count: int,
        num_factors: int = 10
    ) -> str:
        """Build prompt for API call"""
        op_descriptions = FactorOperators.get_operator_descriptions()

        prompt = f"""你是专业的量化金融因子生成专家，需要基于以下规则生成高收益的量化因子：

# 核心规则
1. 可用特征（仅使用这些特征）：
{', '.join(features)}

2. 可用算子（仅使用这些算子）：
{chr(10).join(op_descriptions)}

3. 组合规则：
- 每个因子表达式至少包含{min_op_count}个算子
- 表达式总长度不超过{max_seq_len}个元素（特征+算子）
- 必须符合金融逻辑（动量、反转、波动率、量价结合等）
- 避免除零错误（除法时可加微小值，如 ret_norm / (vol_chg_norm + 1e-6)）
- 表达式必须可被Python执行，且返回一维数值序列

# 金融逻辑要求
- 动量因子：捕捉价格趋势延续性（如 ret_norm + mom5_norm）
- 反转因子：捕捉价格均值回归（如 NEG(ret5_norm) * vol_chg_norm）
- 量价因子：结合成交量和价格变化（如 vol_chg_norm * trend_norm）
- 波动率因子：基于价格波动的因子（如 ABS(DELTA(ret_norm)) + vol_chg_norm）

# 输出要求
请生成 {num_factors} 个不同的因子表达式。
严格按照以下JSON格式输出，仅返回JSON，无其他内容：
{{
  "generated_factors": [
    "ret_norm + vol_chg_norm * trend_norm",
    "ABS(ret5_norm) - DELTA(vol_chg_norm)",
    "mom5_norm / (trend_norm + 1e-6) * ret_norm"
  ]
}}
"""
        return prompt

    def generate_factors_with_deepseek(
        self,
        features: List[str],
        max_seq_len: int = 6,
        min_op_count: int = 2,
        num_factors: int = 10
    ) -> List[str]:
        """
        Generate factors using DeepSeek API

        Args:
            features: List of feature names
            max_seq_len: Maximum sequence length
            min_op_count: Minimum operator count
            num_factors: Number of factors to generate

        Returns:
            List of valid factor expressions
        """
        logger.info(f"开始生成因子，特征: {features}")

        # Build prompt
        prompt = self._build_prompt(features, max_seq_len, min_op_count, num_factors)

        # Call API
        logger.info("调用DeepSeek生成因子组合...")
        api_response = self.api_client.call_api(prompt)

        if not api_response:
            logger.error("DeepSeek API调用失败，使用本地生成器")
            return self.generate_factors_local(features, num_factors)

        # Parse generated factors
        parsed_result = self.parse_factor_expressions(api_response, features)
        generated_factors = parsed_result.get("generated_factors", [])

        # Validate factor expressions
        valid_factors = []
        for expr in generated_factors:
            # Replace operator names with actual symbols
            clean_expr = expr.replace("ADD", "+").replace("SUB", "-").replace("MUL", "*").replace("DIV", "/")

            # Validate expression using FactorCalculator (handles custom operators)
            try:
                test_data = {f: np.random.rand(100) for f in features}
                # Use FactorCalculator to safely evaluate the expression
                test_result = self.calculator.calculate_factor_value(clean_expr, test_data)
                # Verify result is valid
                if test_result is not None and isinstance(test_result, np.ndarray) and len(test_result) == 100:
                    valid_factors.append(clean_expr)
            except (ValueError, TypeError, NameError, ZeroDivisionError) as e:
                logger.warning(f"无效因子表达式：{expr}，错误：{type(e).__name__}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"因子验证时发生未预期错误：{expr}，错误：{e}")
                continue

        logger.info(f"DeepSeek生成了{len(valid_factors)}个有效因子表达式")
        return valid_factors[:num_factors] if valid_factors else self.generate_factors_local(features, num_factors)

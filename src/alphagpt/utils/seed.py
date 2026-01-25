"""Random seed utilities"""

import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"随机种子已设置为: {seed}")

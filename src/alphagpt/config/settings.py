"""Configuration settings for AlphaGPT"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List
import yaml

from alphagpt.utils.env_loader import get_env_or_config


@dataclass
class Config:
    """Main configuration class"""

    # Tushare configuration
    tushare_token: Optional[str] = None  # Tushare API Token

    # Data configuration - qlib format
    # 股票代码格式: 代码.交易所，例如 SZ000001 (平安银行)
    codes: List[str] = field(default_factory=lambda: ['SZ000001'])  # 股票代码列表
    start_date: str = '20200101'  # 开始日期 YYYYMMDD
    end_date: str = '20231231'    # 结束日期 YYYYMMDD
    test_end: str = '20231201'    # 测试集结束日期 YYYYMMDD

    # Factor generation configuration
    max_seq_len: int = 6
    min_op_count: int = 2
    num_factors: int = 10

    # DeepSeek API configuration
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_timeout: int = 60
    temperature: float = 0.7
    use_api: bool = True  # 是否使用 API（False 则使用本地生成）

    # gemini API config
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    gemini_temperature: float = 0.7
    gemini_proxy: Optional[str] = None

    # Backtest configuration
    position_scale: float = 0.3

    # Logging configuration
    log_level: int = logging.INFO
    log_file: str = 'logs/alphagpt.log'

    # Random seed
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, config_path: str = 'config.yaml') -> 'Config':
        """
        从 YAML 配置文件加载配置

        Args:
            config_path: 配置文件路径

        Returns:
            Config 实例
        """
        if not os.path.exists(config_path):
            logging.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return cls()

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config is None:
                logging.warning(f"配置文件 {config_path} 为空，使用默认配置")
                return cls()

            # 从 YAML 结构中提取配置
            config_dict = {}

            # Tushare 配置 (优先使用环境变量)
            if 'tushare' in yaml_config:
                tushare_token = yaml_config['tushare'].get('token')
                config_dict['tushare_token'] = get_env_or_config('TUSHARE_TOKEN', tushare_token)

            # 数据配置
            if 'data' in yaml_config:
                data_config = yaml_config['data']
                # 支持 codes (多标的) 或 code (单标的)
                if 'codes' in data_config:
                    config_dict['codes'] = data_config['codes']
                elif 'code' in data_config:
                    config_dict['codes'] = [data_config['code']]
                else:
                    config_dict['codes'] = ['SZ000001']

                config_dict['start_date'] = data_config.get('start_date', '20200101')
                config_dict['end_date'] = data_config.get('end_date', '20231231')
                config_dict['test_end'] = data_config.get('test_end', '20231201')

            # 因子配置
            if 'factor' in yaml_config:
                factor_config = yaml_config['factor']
                config_dict['max_seq_len'] = factor_config.get('max_seq_len', 6)
                config_dict['min_op_count'] = factor_config.get('min_op_count', 2)
                config_dict['num_factors'] = factor_config.get('num_factors', 10)

            # DeepSeek API 配置 (优先使用环境变量)
            if 'deepseek' in yaml_config:
                api_config = yaml_config['deepseek']
                api_key = api_config.get('api_key')
                config_dict['api_key'] = get_env_or_config('DEEPSEEK_API_KEY', api_key)
                config_dict['base_url'] = api_config.get('base_url', 'https://api.deepseek.com')
                config_dict['model'] = api_config.get('model', 'deepseek-chat')
                config_dict['api_timeout'] = api_config.get('timeout', 60)
                config_dict['temperature'] = api_config.get('temperature', 0.7)

            # gemini配置
            if 'gemini' in yaml_config:
                gemini_config = yaml_config['gemini']
                print('设置gemini api key: {}'.format(get_env_or_config('GEMINI_API_KEY', gemini_config.get('api_key'))))
                config_dict['gemini_api_key'] = get_env_or_config('GEMINI_API_KEY', gemini_config.get('api_key'))
                config_dict['gemini_model'] = gemini_config.get('model')
                config_dict['gemini_temperature'] = gemini_config.get('temperature', 0.7)
                config_dict['gemini_proxy'] = gemini_config.get('proxy', None)

            # 回测配置
            if 'backtest' in yaml_config:
                config_dict['position_scale'] = yaml_config['backtest'].get('position_scale', 0.3)

            # 日志配置
            if 'logging' in yaml_config:
                log_config = yaml_config['logging']
                log_level_str = log_config.get('level', 'INFO')
                config_dict['log_level'] = getattr(logging, log_level_str.upper(), logging.INFO)
                config_dict['log_file'] = log_config.get('log_file', 'logs/alphagpt.log')

            # 随机种子
            config_dict['random_seed'] = yaml_config.get('random_seed', 42)

            return cls(**config_dict)

        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}，使用默认配置")
            return cls()

    def to_yaml(self, config_path: str = 'config.yaml'):
        """
        将当前配置保存到 YAML 文件

        Args:
            config_path: 配置文件路径
        """
        yaml_config = {
            'tushare': {
                'token': self.tushare_token or 'your_tushare_token_here',
                'timeout': 30
            },
            'data': {
                'ts_codes': self.ts_codes,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'test_end': self.test_end
            },
            'factor': {
                'max_seq_len': self.max_seq_len,
                'min_op_count': self.min_op_count,
                'num_factors': self.num_factors
            },
            'deepseek': {
                'api_key': self.api_key or 'your_deepseek_api_key_here',
                'base_url': self.base_url,
                'model': self.model,
                'timeout': self.api_timeout,
                'temperature': self.temperature
            },
            'backtest': {
                'position_scale': self.position_scale
            },
            'logging': {
                'level': logging.getLevelName(self.log_level),
                'log_file': self.log_file
            },
            'random_seed': self.random_seed
        }

        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else '.', exist_ok=True)

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logging.info(f"配置已保存到 {config_path}")

"""Custom exception classes for AlphaGPT"""


class AlphaGPTException(Exception):
    """Base exception class for AlphaGPT"""
    pass


class DataLoadError(AlphaGPTException):
    """Exception raised when data loading fails"""
    pass


class FactorGenerationError(AlphaGPTException):
    """Exception raised when factor generation fails"""
    pass


class ConfigurationError(AlphaGPTException):
    """Exception raised when configuration is invalid"""
    pass


class ValidationError(AlphaGPTException):
    """Exception raised when data validation fails"""
    pass


class BacktestError(AlphaGPTException):
    """Exception raised when backtesting fails"""
    pass

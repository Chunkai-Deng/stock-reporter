"""Unified configuration loader for stock-reporter.

Reads from environment variables, falling back to .env file.
Provides a typed Config dataclass and a singleton get_config().
"""

import os
from dataclasses import dataclass, field


def _load_env() -> dict:
    """Load key=value pairs from the .env file next to the project root."""
    env = {}
    # Project root is parent of lib/
    lib_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(lib_dir)
    env_path = os.path.join(project_dir, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    env[key.strip()] = val.strip()
        except Exception:
            pass
    return env


_ENV = _load_env()


def _env_or(key: str, default: str = "") -> str:
    """Read config from environment variable, falling back to .env, then default."""
    return os.environ.get(key) or _ENV.get(key) or default


@dataclass
class Config:
    """Typed configuration for the stock reporter system."""

    # WeCom
    webhook_url: str = field(default_factory=lambda: _env_or("WECOM_WEBHOOK_URL", ""))

    # Stock list
    stock_codes_raw: str = field(default_factory=lambda: _env_or("STOCK_CODES", "600519,000858"))
    stock_codes: list = field(default_factory=list)

    # DeepSeek
    deepseek_api_key: str = field(default_factory=lambda: _env_or("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: _env_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    deepseek_model: str = field(
        default_factory=lambda: _env_or("DEEPSEEK_MODEL", "deepseek-v4-pro")
    )

    # Daemon
    report_interval_minutes: int = field(
        default_factory=lambda: int(_env_or("REPORT_INTERVAL_MINUTES", "60"))
    )

    # Board filter (pre-screening)
    # main=主板, chinext=创业板, star=科创板, bse=北交所 — comma-separated
    stock_boards: str = field(
        default_factory=lambda: _env_or("STOCK_BOARDS", "main,chinext")
    )
    exclude_st: bool = field(
        default_factory=lambda: _env_or("EXCLUDE_ST", "true").lower() == "true"
    )

    # Feature flags
    enable_enhanced_analysis: bool = field(
        default_factory=lambda: _env_or("ENABLE_ENHANCED_ANALYSIS", "false").lower() == "true"
    )
    screening_enabled: bool = field(
        default_factory=lambda: _env_or("SCREENING_ENABLED", "false").lower() == "true"
    )

    def __post_init__(self):
        self.stock_codes = [c.strip() for c in self.stock_codes_raw.split(",") if c.strip()]

    # ── Board prefix helpers ───────────────────────────────────────────

    _BOARD_PREFIX_MAP: dict = field(default_factory=lambda: {
        "main": ("60", "00"),
        "chinext": ("30",),
        "star": ("688",),
        "bse": ("8", "4"),
    }, repr=False, init=False)

    @property
    def allowed_prefixes(self) -> tuple[str, ...]:
        """将 STOCK_BOARDS 转成股票代码前缀白名单。"""
        prefixes = []
        for board in [b.strip() for b in self.stock_boards.split(",") if b.strip()]:
            prefixes.extend(self._BOARD_PREFIX_MAP.get(board, []))
        return tuple(prefixes) if prefixes else ("60", "00", "30")

    @property
    def boards_slug(self) -> str:
        """boards 短标识用于缓存文件名，如 'main_chinext' 或 'main'。"""
        boards = sorted([b.strip() for b in self.stock_boards.split(",") if b.strip()])
        return "_".join(boards) if boards else "main_chinext"


_config_singleton: Config | None = None


def get_config() -> Config:
    """Return the singleton Config instance."""
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = Config()
    return _config_singleton

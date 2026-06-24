"""Configuration loader — reads from .env and env vars."""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    wecom_webhook_url: str = ""
    stock_codes: List[str] = field(default_factory=list)
    report_interval_minutes: int = 60
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    enable_enhanced_analysis: bool = False
    screening_enabled: bool = False
    stock_boards: List[str] = field(default_factory=lambda: ["main", "chinext"])
    exclude_st: bool = True
    min_turnover: float = 100_000_000  # 1亿

    @property
    def allowed_prefixes(self) -> List[str]:
        """Map board names -> stock code prefixes."""
        mapping = {
            "main": ["sh60", "sz00"],
            "chinext": ["sz30"],
            "star": ["sh68"],
            "bse": ["bj43", "bj83", "bj87", "bj89"],
        }
        prefixes = []
        for board in self.stock_boards:
            prefixes.extend(mapping.get(board, []))
        return prefixes

    @property
    def boards_slug(self) -> str:
        """Stable, sorted identifier for cache sharding."""
        return "_".join(sorted(self.stock_boards))

    @property
    def deepseek_available(self) -> bool:
        return bool(self.deepseek_api_key)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes")


def load_config() -> Config:
    """Load configuration from .env file (if present) and environment variables."""
    # Try to load .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value

    def get(key: str, default: str = "") -> str:
        return os.environ.get(key, default)

    return Config(
        wecom_webhook_url=get("WECOM_WEBHOOK_URL"),
        stock_codes=[c.strip() for c in get("STOCK_CODES").split(",") if c.strip()],
        report_interval_minutes=int(get("REPORT_INTERVAL_MINUTES", "60")),
        deepseek_api_key=get("DEEPSEEK_API_KEY"),
        deepseek_model=get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        deepseek_base_url=get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        enable_enhanced_analysis=_parse_bool(get("ENABLE_ENHANCED_ANALYSIS", "false")),
        screening_enabled=_parse_bool(get("SCREENING_ENABLED", "false")),
        stock_boards=[b.strip() for b in get("STOCK_BOARDS", "main,chinext").split(",") if b.strip()],
        exclude_st=_parse_bool(get("EXCLUDE_ST", "true")),
        min_turnover=float(get("MIN_TURNOVER", "100000000")),
    )

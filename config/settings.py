import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AlpacaSettings:
    paper_api_key: str
    paper_api_secret: str


@dataclass(frozen=True)
class B2Settings:
    endpoint_url: str
    region: str
    bucket_name: str
    key_prefix: str
    access_key_id: str
    secret_access_key: str


@dataclass(frozen=True)
class RuntimeSettings:
    enable_torch_compile: bool


@dataclass(frozen=True)
class Settings:
    alpaca: AlpacaSettings
    b2: B2Settings
    runtime: RuntimeSettings

    @classmethod
    def load(
        cls,
        settings_path: str | Path = _REPO_ROOT / "settings.yaml",
        env_path: str | Path = _REPO_ROOT / ".env",
        environ: Mapping[str, str] | None = None,
    ) -> "Settings":
        settings_path = Path(settings_path)
        if not settings_path.is_file():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")

        with settings_path.open(encoding="utf-8") as settings_file:
            raw_settings = yaml.safe_load(settings_file) or {}
        if not isinstance(raw_settings, dict):
            raise ValueError("settings.yaml must contain a YAML mapping")

        _reject_unknown_keys(raw_settings, {"b2", "runtime"}, "settings")
        b2 = _required_mapping(raw_settings, "b2")
        runtime = _required_mapping(raw_settings, "runtime")
        _reject_unknown_keys(
            b2,
            {"endpoint_url", "region", "bucket_name", "key_prefix"},
            "b2",
        )
        _reject_unknown_keys(
            runtime,
            {"enable_torch_compile"},
            "runtime",
        )

        secret_values = {
            key: value
            for key, value in dotenv_values(env_path).items()
            if value is not None
        }
        secret_values.update(os.environ if environ is None else environ)

        return cls(
            alpaca=AlpacaSettings(
                paper_api_key=_required_secret(
                    secret_values,
                    "ALPACA_PAPER_API_KEY",
                ),
                paper_api_secret=_required_secret(
                    secret_values,
                    "ALPACA_PAPER_API_SECRET",
                ),
            ),
            b2=B2Settings(
                endpoint_url=_required_string(b2, "endpoint_url", "b2"),
                region=_required_string(b2, "region", "b2"),
                bucket_name=_required_string(b2, "bucket_name", "b2"),
                key_prefix=_required_string(b2, "key_prefix", "b2"),
                access_key_id=_required_secret(
                    secret_values,
                    "B2_ACCESS_KEY_ID",
                ),
                secret_access_key=_required_secret(
                    secret_values,
                    "B2_SECRET_ACCESS_KEY",
                ),
            ),
            runtime=RuntimeSettings(
                enable_torch_compile=_required_bool(
                    runtime,
                    "enable_torch_compile",
                    "runtime",
                ),
            ),
        )


def _required_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"settings.{key} must be a YAML mapping")
    return value


def _required_string(
    data: Mapping[str, Any],
    key: str,
    section: str,
) -> str:
    value = data.get(key)
    if (
        not isinstance(value, str)
        or not value.strip()
        or value.startswith("REPLACE_")
    ):
        raise ValueError(f"settings.{section}.{key} must be configured")
    return value


def _required_bool(
    data: Mapping[str, Any],
    key: str,
    section: str,
) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"settings.{section}.{key} must be a boolean")
    return value


def _required_secret(values: Mapping[str, str], key: str) -> str:
    value = values.get(key)
    if not value:
        raise ValueError(f"Missing required secret in .env: {key}")
    return value


def _reject_unknown_keys(
    data: Mapping[str, Any],
    expected: set[str],
    section: str,
) -> None:
    unknown = set(data) - expected
    if unknown:
        raise ValueError(
            f"Unknown {section} setting(s): {', '.join(sorted(unknown))}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.load()

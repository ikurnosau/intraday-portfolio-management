from pathlib import Path

import pytest

from config.settings import (
    AlpacaSettings,
    B2Settings,
    RuntimeSettings,
    Settings,
    get_settings,
)


VALID_SETTINGS = """
b2:
  endpoint_url: https://s3.us-east-005.backblazeb2.com
  region: us-east-005
  bucket_name: market-data
  key_prefix: alpaca
runtime:
  enable_torch_compile: false
"""

VALID_SECRETS = """
ALPACA_PAPER_API_KEY=alpaca-key
ALPACA_PAPER_API_SECRET=alpaca-secret
B2_ACCESS_KEY_ID=b2-key
B2_SECRET_ACCESS_KEY=b2-secret
"""


def write_config_files(tmp_path: Path) -> tuple[Path, Path]:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(VALID_SETTINGS, encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text(VALID_SECRETS, encoding="utf-8")
    return settings_path, env_path


def test_load_combines_yaml_settings_and_env_secrets(tmp_path):
    settings_path, env_path = write_config_files(tmp_path)

    settings = Settings.load(settings_path, env_path, environ={})

    assert settings.alpaca.paper_api_key == "alpaca-key"
    assert settings.b2.bucket_name == "market-data"
    assert settings.b2.access_key_id == "b2-key"
    assert settings.runtime.enable_torch_compile is False


def test_environment_overrides_dotenv_secrets(tmp_path):
    settings_path, env_path = write_config_files(tmp_path)

    settings = Settings.load(
        settings_path,
        env_path,
        environ={"ALPACA_PAPER_API_KEY": "override-key"},
    )

    assert settings.alpaca.paper_api_key == "override-key"


def test_missing_secret_has_actionable_error(tmp_path):
    settings_path, env_path = write_config_files(tmp_path)
    env_path.write_text(
        VALID_SECRETS.replace("B2_SECRET_ACCESS_KEY=b2-secret\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="B2_SECRET_ACCESS_KEY"):
        Settings.load(settings_path, env_path, environ={})


def test_invalid_yaml_setting_type_is_rejected(tmp_path):
    settings_path, env_path = write_config_files(tmp_path)
    settings_path.write_text(
        VALID_SETTINGS.replace(
            "enable_torch_compile: false",
            'enable_torch_compile: "false"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        Settings.load(settings_path, env_path, environ={})


def test_missing_settings_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Settings file not found"):
        Settings.load(
            tmp_path / "missing.yaml",
            tmp_path / ".env",
            environ={},
        )


def test_get_settings_is_cached(monkeypatch):
    expected = Settings(
        alpaca=AlpacaSettings("alpaca-key", "alpaca-secret"),
        b2=B2Settings(
            endpoint_url="https://example.com",
            region="region",
            bucket_name="bucket",
            key_prefix="alpaca",
            access_key_id="b2-key",
            secret_access_key="b2-secret",
        ),
        runtime=RuntimeSettings(enable_torch_compile=False),
    )
    monkeypatch.setattr(
        Settings,
        "load",
        classmethod(lambda cls: expected),
    )
    get_settings.cache_clear()

    try:
        assert get_settings() is expected
        assert get_settings() is expected
    finally:
        get_settings.cache_clear()

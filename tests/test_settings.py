import pytest

from config.settings import Settings


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_user_cloud_variable_names_load_without_exposing_keys(tmp_path, monkeypatch, encoding) -> None:
    for name in (
        "NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY", "NIM_API_KEY", "Nvidia",
        "NVIDIA_MODEL", "NVIDIA_MODEL_NAME", "NIM_MODEL", "NIM_MODEL_NAME", "Nmodel",
        "GROQ_API_KEY", "Groq", "GROQ_MODEL", "GROQ_MODEL_NAME", "GModel",
    ):
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        'Nvidia="test-nvidia-secret"\nNmodel=nvidia/test-nemotron\n'
        "Groq='test-groq-secret'\nGModel=qwen/test-model\n",
        encoding=encoding,
    )
    # Restore the environment entries that the dotenv loader adds as well.
    for name in ("Nvidia", "Nmodel", "Groq", "GModel"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)

    settings = Settings.from_env(str(dotenv))

    assert settings.nvidia_api_key == "test-nvidia-secret"
    assert settings.groq_api_key == "test-groq-secret"
    assert settings.nvidia_model == "nvidia/test-nemotron"
    assert settings.groq_model == "qwen/test-model"
    for exported in (repr(settings), repr(settings.as_dict())):
        assert "test-nvidia-secret" not in exported
        assert "test-groq-secret" not in exported


def test_standard_cloud_variables_take_precedence_over_user_aliases(tmp_path, monkeypatch) -> None:
    for name, value in {
        "NVIDIA_API_KEY": "preferred-nvidia-key", "NVIDIA_MODEL": "preferred-nemotron",
        "Nvidia": "alias-nvidia-key", "Nmodel": "alias-nemotron",
        "GROQ_API_KEY": "preferred-groq-key", "GROQ_MODEL": "preferred-qwen",
        "Groq": "alias-groq-key", "GModel": "alias-qwen",
    }.items():
        monkeypatch.setenv(name, value)

    settings = Settings.from_env(str(tmp_path / "absent.env"))

    assert settings.nvidia_api_key == "preferred-nvidia-key"
    assert settings.nvidia_model == "preferred-nemotron"
    assert settings.groq_api_key == "preferred-groq-key"
    assert settings.groq_model == "preferred-qwen"

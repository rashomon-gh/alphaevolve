import pytest

from alphaevolve.config import deep_merge, load_run_config, load_yaml


def test_env_interpolation_with_default(tmp_path, monkeypatch):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("url: ${TEST_AE_URL:-http://fallback:1/v1}\nname: plain\n")
    assert load_yaml(cfg_file)["url"] == "http://fallback:1/v1"
    monkeypatch.setenv("TEST_AE_URL", "http://real:2/v1")
    assert load_yaml(cfg_file)["url"] == "http://real:2/v1"


def test_env_interpolation_missing_raises(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("key: ${DEFINITELY_NOT_SET_AE}\n")
    with pytest.raises(KeyError):
        load_yaml(cfg_file)


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 20}, "c": 4}
    assert deep_merge(base, override) == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}


def test_base_inheritance_anchors_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "base.yaml").write_text("task: {name: t}\nlimits: {max_samples: 100}\n")
    (tmp_path / "sub" / "child.yaml").write_text(
        "base: ../base.yaml\nlimits: {max_samples: 5}\nablation: {no_meta: true}\n"
    )
    config, anchor = load_run_config(tmp_path / "sub" / "child.yaml")
    assert anchor == tmp_path  # paths resolve relative to the base file
    assert config["limits"]["max_samples"] == 5
    assert config["task"]["name"] == "t"
    assert config["ablation"]["no_meta"] is True

"""Registry and install() plans: pure Python, no framework needed."""
import os

import pytest

import ai_search_plugin as asp
from ai_search_plugin import registry as R
from ai_search_plugin.frameworks import SGL_BOOT_DIR


def test_registry_shape():
    names = [c.name for c in R.COMPONENTS]
    assert len(names) == len(set(names))
    for c in R.COMPONENTS:
        assert c.category in (R.MODEL_AGNOSTIC, R.MODEL_FAMILY), c
        assert c.status in (R.PORTED, R.STUB), c
        assert c.frameworks and set(c.frameworks) <= set(R.FRAMEWORKS), c
        assert c.summary and len(c.summary) < 200, c
        if c.status == R.STUB:
            assert not c.default, f"stub {c.name} cannot be in the default set"
        if c.category == R.MODEL_FAMILY:
            assert c.family, c


def test_every_module_importable_without_frameworks():
    import importlib
    for c in R.COMPONENTS:
        for m in c.modules:
            if m.endswith("sgl_boot"):
                assert os.path.isfile(os.path.join(SGL_BOOT_DIR, "sitecustomize.py"))
                continue
            if m == "response_logits":   # needs torch (GPU code); import only when torch is present
                pytest.importorskip("torch")
            importlib.import_module("ai_search_plugin." + m)


def test_stub_install_raises():
    import importlib
    from ai_search_plugin.stubs import NotPortedError
    for c in R.list_components(status=R.STUB):
        mod = importlib.import_module("ai_search_plugin." + c.modules[0])
        with pytest.raises(NotPortedError):
            mod.install()


def test_resolve_rules():
    d = [c.name for c in R.resolve("slime", "default")]
    assert "ticket_admission" in d and "kvfresh" not in d and "prefix_tree" not in d
    assert [c.name for c in R.resolve("slime", "default,-tool_hedge")] == [n for n in d if n != "tool_hedge"]
    assert R.resolve("verl", "none") == []
    with pytest.raises(ValueError):
        R.resolve("slime", ["gdn_cudnn"])              # stub
    with pytest.raises(ValueError):
        R.resolve("verl", ["ticket_admission"])        # slime only
    with pytest.raises(ValueError):
        R.resolve("verl", ["verl_kvfresh", "verl_inplace"])   # conflict
    a = [c.name for c in R.resolve("verl", "all")]
    assert not ({"verl_kvfresh", "verl_inplace"} <= set(a))
    with pytest.raises(KeyError):
        R.resolve("slime", ["no_such_component"])


def test_slime_plan_default():
    env = {}
    p = asp.install("slime", "default", options={"toolcache_generate": "my_agent.generate",
                                                 "tool_targets": ["my_agent.search"]}, environ=env)
    e = p.env
    assert "--custom-megatron-init-path" in p.args and "ai_search_plugin.frameworks.slime.megatron_init" in p.args
    i = p.args.index("--rollout-function-path")
    assert p.args[i + 1] == "ai_search_plugin.adapters.slime_rshard.generate_rollout_fully_async"
    assert e["ASRT_RS_INNER"] == "ai_search_plugin.adapters.slime_admit.generate_rollout_fully_async"
    assert e["ASRT_ADMIT_INNER"] == "ai_search_plugin.adapters.slime_prefetch.generate_rollout_fully_async"
    assert e["ASRT_MAX_STALENESS"] == "2" and e["ASRT_POOL"] == "auto"
    assert set(e["ASRT_SGL_HOOKS"].split(",")) == {"nosync", "fastload", "fastload_check"}
    assert e["ASRT_WEIGHTSYNC"] == "pipeline,flatten,parsync,profile"
    assert e["ASRT_PREFETCH"] == "overlap,profile,r3c" and e["ASRT_R3CHECK"] == "fast"
    assert e["ASRT_JITFIX"] == "eager"
    assert "response_logits" in e["ASRT_CHAIN_MEGATRON_INIT"]
    assert e["ASRT_TOOLCACHE_GENERATE"] == "my_agent.generate" and e["ASRT_TOOL_HEDGE_TARGETS"] == "my_agent.search"
    assert p.launcher[-1] == "ai_search_plugin.adapters.slime_train"
    assert env["PYTHONPATH"].split(":")[0] == SGL_BOOT_DIR
    assert "export ASRT_MAX_STALENESS=2" in p.shell()


def test_slime_plan_skips_unconfigured():
    p = asp.plan("slime", "default")      # no tool functions given
    assert "toolcache" in p.skipped and "tool_hedge" in p.skipped
    assert "--custom-generate-function-path" not in p.args
    p = asp.plan("slime", ["weight_sync"])
    assert p.env["ASRT_POOL"] == "0" and "--rollout-function-path" not in p.args and not p.launcher
    p = asp.plan("slime", "none")
    assert p.env["ASRT_WEIGHTSYNC"] == "off" and "ASRT_SGL_HOOKS" not in p.env and not p.pythonpath_prepend


def test_verl_plan_default():
    env = {}
    p = asp.install("verl", "default", environ=env)
    feats = env["ASRT_VERL"].split(",")
    assert {"sglhooks", "inplace", "ceiling", "toolcache", "response_logits"} <= set(feats)
    assert "kvfresh" not in feats and env["ASRT_MAX_STALENESS"] == "2"
    assert "ai_search_plugin.verl_plugin" in env["VERL_USE_EXTERNAL_MODULES"].split(",")
    assert env["ASRT_SGL_HOOKS"] == "nosync,fastload"
    p = asp.plan("verl", ["verl_kvfresh"])
    assert p.env["ASRT_VERL"] == "kvfresh" and p.env["ASRT_VERL_KVFRESH"] == "1"


def test_cli_list_and_plan(capsys):
    from ai_search_plugin.__main__ import main
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ticket_admission" in out and "stub" in out
    assert main(["plan", "verl", "verl_ceiling", "max_staleness=3"]) == 0
    assert "ASRT_MAX_STALENESS=3" in capsys.readouterr().out

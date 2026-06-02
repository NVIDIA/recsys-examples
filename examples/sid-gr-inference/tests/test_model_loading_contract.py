# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json

import pytest
from gr_inference.gr_models import (
    HFCheckpointLoader,
    TensorLoadRequest,
    resolve_model_dir,
)
from gr_inference.gr_models.qwen3 import (
    Qwen3GRConfig,
    Qwen3HFAdapter,
    get_qwen3_variant,
    identify_qwen3_variant,
    resolve_qwen3_model_dir,
)


class FakeTensor:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape

    @classmethod
    def concat(cls, tensors: list["FakeTensor"], *, dim: int) -> "FakeTensor":
        shape = list(tensors[0].shape)
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        name = "cat(" + ",".join(tensor.name for tensor in tensors) + ")"
        return cls(name, tuple(shape))


def write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_minimal_checkpoint_dir(path) -> None:
    write_json(path / "config.json", qwen3_config(num_layers=1))
    write_json(
        path / "model.safetensors.index.json",
        {"metadata": {}, "weight_map": {}},
    )


def qwen3_config(num_layers: int = 2) -> dict:
    return {
        "_name_or_path": "Qwen/Qwen3-0.6B",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_hidden_layers": num_layers,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }


def qwen3_1_7b_config(num_layers: int = 28) -> dict:
    return {
        "_name_or_path": "Qwen/Qwen3-1.7B",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_hidden_layers": num_layers,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }


def make_weight_map(adapter: Qwen3HFAdapter) -> dict[str, str]:
    names = adapter.weight_names()
    weight_names = list(names.required()) + list(names.optional())
    return {name: "model-00001-of-00001.safetensors" for name in weight_names}


def make_sharded_weight_map(adapter: Qwen3HFAdapter) -> dict[str, str]:
    names = adapter.weight_names()
    weight_map: dict[str, str] = {
        names.embed_tokens: "model-00001-of-00002.safetensors",
        names.final_norm: "model-00002-of-00002.safetensors",
    }
    if names.lm_head is not None:
        weight_map[names.lm_head] = "model-00002-of-00002.safetensors"
    for layer_idx, layer in enumerate(names.layers):
        shard = (
            "model-00001-of-00002.safetensors"
            if layer_idx == 0
            else "model-00002-of-00002.safetensors"
        )
        for tensor_name in layer.required() + layer.optional():
            weight_map[tensor_name] = shard
    return weight_map


def test_hf_checkpoint_loader_reads_config_and_index(tmp_path) -> None:
    config = Qwen3GRConfig.from_hf_config(qwen3_config())
    adapter = Qwen3HFAdapter(config)

    write_json(tmp_path / "config.json", qwen3_config())
    write_json(
        tmp_path / "model.safetensors.index.json",
        {"metadata": {}, "weight_map": make_weight_map(adapter)},
    )

    manifest = HFCheckpointLoader(tmp_path).manifest()

    assert manifest.config["architectures"] == ["Qwen3ForCausalLM"]
    assert manifest.has_tensor("model.embed_tokens.weight")
    assert manifest.weight_files == ("model-00001-of-00001.safetensors",)


def test_qwen3_config_from_hf_config() -> None:
    config = Qwen3GRConfig.from_hf_config(qwen3_config(num_layers=28))
    variant = identify_qwen3_variant(config)

    assert config.model_name == "Qwen/Qwen3-0.6B"
    assert config.num_layers == 28
    assert config.hidden_size == 1024
    assert config.num_attention_heads == 16
    assert config.num_kv_heads == 8
    assert config.head_dim == 128
    assert config.intermediate_size == 3072
    assert config.vocab_size == 151936
    assert variant is not None
    assert variant.canonical_name == "qwen3-0.6b"


def test_qwen3_1_7b_config_from_hf_config_and_variant_metadata() -> None:
    config = Qwen3GRConfig.from_hf_config(qwen3_1_7b_config())
    variant = identify_qwen3_variant(config)
    adapter = Qwen3HFAdapter(config)

    assert config.model_name == "Qwen/Qwen3-1.7B"
    assert config.num_layers == 28
    assert config.hidden_size == 2048
    assert config.intermediate_size == 6144
    assert config.num_attention_heads == 16
    assert config.num_kv_heads == 8
    assert config.head_dim == 128
    assert config.q_size == 2048
    assert config.kv_size == 1024
    assert config.qkv_size == 4096
    assert config.gate_up_size == 12288
    assert config.vocab_size == 151936
    assert config.tie_word_embeddings is True
    assert variant is not None
    assert variant.canonical_name == "qwen3-1.7b"
    assert adapter.weight_names().lm_head is None


def test_qwen3_variant_materializes_1_7b_gr_config() -> None:
    config = get_qwen3_variant("qwen3-1.7b").to_gr_config(max_beam_width=256)

    assert config.model_name == "qwen3-1.7b"
    assert config.hidden_size == 2048
    assert config.intermediate_size == 6144
    assert config.qkv_size == 4096
    assert config.gate_up_size == 12288
    assert config.max_beam_width == 256
    assert config.tie_word_embeddings is True


def test_qwen3_adapter_generates_hf_tensor_names() -> None:
    adapter = Qwen3HFAdapter(Qwen3GRConfig.from_hf_config(qwen3_config(num_layers=2)))
    names = adapter.weight_names()
    layer0 = names.layers[0]

    assert names.embed_tokens == "model.embed_tokens.weight"
    assert names.final_norm == "model.norm.weight"
    assert names.lm_head == "lm_head.weight"
    assert layer0.q_proj == "model.layers.0.self_attn.q_proj.weight"
    assert layer0.k_proj == "model.layers.0.self_attn.k_proj.weight"
    assert layer0.v_proj == "model.layers.0.self_attn.v_proj.weight"
    assert layer0.o_proj == "model.layers.0.self_attn.o_proj.weight"
    assert layer0.q_norm == "model.layers.0.self_attn.q_norm.weight"
    assert layer0.k_norm == "model.layers.0.self_attn.k_norm.weight"
    assert len(names.layers) == 2


def test_qwen3_adapter_validates_required_tensors(tmp_path) -> None:
    config = Qwen3GRConfig.from_hf_config(qwen3_config(num_layers=1))
    adapter = Qwen3HFAdapter(config)
    weight_map = make_weight_map(adapter)
    del weight_map["model.layers.0.self_attn.q_proj.weight"]

    write_json(tmp_path / "config.json", qwen3_config(num_layers=1))
    write_json(
        tmp_path / "model.safetensors.index.json",
        {"metadata": {}, "weight_map": weight_map},
    )
    manifest = HFCheckpointLoader(tmp_path).manifest()

    with pytest.raises(KeyError, match="q_proj"):
        adapter.validate_manifest(manifest)


def test_tied_embeddings_make_lm_head_optional() -> None:
    config_dict = qwen3_config(num_layers=1)
    config_dict["tie_word_embeddings"] = True
    adapter = Qwen3HFAdapter(Qwen3GRConfig.from_hf_config(config_dict))

    names = adapter.weight_names()

    assert names.lm_head is None
    assert "lm_head.weight" in names.optional()


def test_qwen3_load_plan_packs_qkv_and_gate_up() -> None:
    adapter = Qwen3HFAdapter(Qwen3GRConfig.from_hf_config(qwen3_config(num_layers=1)))

    plan = adapter.load_plan()
    requests = {request.logical_name: request for request in plan.requests}

    qkv = requests["layers.0.self_attn.qkv_proj.weight"]
    assert qkv.transform == "concat"
    assert qkv.dim == 0
    assert qkv.source_names == (
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    )

    gate_up = requests["layers.0.mlp.gate_up_proj.weight"]
    assert gate_up.transform == "concat"
    assert gate_up.source_names == (
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    )


def test_load_plan_groups_required_tensors_by_file(tmp_path) -> None:
    adapter = Qwen3HFAdapter(Qwen3GRConfig.from_hf_config(qwen3_config(num_layers=2)))
    write_json(tmp_path / "config.json", qwen3_config(num_layers=2))
    write_json(
        tmp_path / "model.safetensors.index.json",
        {"metadata": {}, "weight_map": make_sharded_weight_map(adapter)},
    )
    manifest = HFCheckpointLoader(tmp_path).manifest()

    grouped = adapter.load_plan().grouped_by_file(manifest)

    assert set(grouped) == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert (
        "model.layers.0.self_attn.q_proj.weight"
        in grouped["model-00001-of-00002.safetensors"]
    )
    assert (
        "model.layers.1.self_attn.q_proj.weight"
        in grouped["model-00002-of-00002.safetensors"]
    )


def test_load_plan_materializes_identity_and_concat() -> None:
    request_identity = TensorLoadRequest(
        "layers.0.input_layernorm.weight",
        ("model.layers.0.input_layernorm.weight",),
    )
    request_concat = TensorLoadRequest(
        "layers.0.self_attn.qkv_proj.weight",
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ),
        transform="concat",
        dim=0,
    )
    from gr_inference.gr_models import CheckpointLoadPlan

    load_plan = CheckpointLoadPlan((request_identity, request_concat))
    tensors = {
        "model.layers.0.input_layernorm.weight": FakeTensor("ln", (32,)),
        "model.layers.0.self_attn.q_proj.weight": FakeTensor("q", (32, 32)),
        "model.layers.0.self_attn.k_proj.weight": FakeTensor("k", (16, 32)),
        "model.layers.0.self_attn.v_proj.weight": FakeTensor("v", (16, 32)),
    }

    materialized = load_plan.materialize(tensors.get)

    assert materialized["layers.0.input_layernorm.weight"].shape == (32,)
    assert materialized["layers.0.self_attn.qkv_proj.weight"].shape == (64, 32)


def test_hf_loader_discovers_unindexed_safetensors(tmp_path) -> None:
    if importlib.util.find_spec("safetensors") is None:
        pytest.skip("safetensors is not installed")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from safetensors.torch import save_file

    write_json(tmp_path / "config.json", qwen3_config(num_layers=1))
    save_file(
        {"model.embed_tokens.weight": torch.zeros(4, 4)},
        tmp_path / "model.safetensors",
    )

    manifest = HFCheckpointLoader(tmp_path).manifest()

    assert manifest.has_tensor("model.embed_tokens.weight")
    assert (
        manifest.tensor_map["model.embed_tokens.weight"].filename == "model.safetensors"
    )


def test_qwen3_model_dir_resolver_defaults_to_1_7b() -> None:
    assert resolve_qwen3_model_dir(env={}) == "models/Qwen3-1.7B"
    assert resolve_qwen3_model_dir(variant="qwen3-0.6b", env={}) == "models/Qwen3-0.6B"
    assert (
        resolve_qwen3_model_dir(
            variant="qwen3-1.7b",
            env={"GR_QWEN3_1_7B_MODEL_DIR": "/models/qwen3-1.7b"},
        )
        == "/models/qwen3-1.7b"
    )


def test_model_resolver_prefers_explicit_model_dir(tmp_path) -> None:
    write_minimal_checkpoint_dir(tmp_path)

    resolved = resolve_model_dir(
        model_dir=tmp_path,
        model="Qwen/Qwen3-1.7B",
        default_model="Qwen/Qwen3-0.6B",
        revision="unused",
    )

    assert resolved == str(tmp_path)


def test_model_resolver_accepts_local_model_path(tmp_path) -> None:
    write_minimal_checkpoint_dir(tmp_path)

    resolved = resolve_model_dir(model=str(tmp_path), default_model="Qwen/Qwen3-1.7B")

    assert resolved == str(tmp_path)


def test_model_resolver_downloads_repo_id_with_revision(tmp_path) -> None:
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    write_minimal_checkpoint_dir(downloaded)
    calls = []

    def fake_downloader(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    resolved = resolve_model_dir(
        model="Qwen/Qwen3-1.7B",
        revision="abc123",
        downloader=fake_downloader,
    )

    assert resolved == str(downloaded)
    assert calls == [{"repo_id": "Qwen/Qwen3-1.7B", "revision": "abc123"}]


def test_model_resolver_downloads_repo_id_to_explicit_dir(tmp_path) -> None:
    local_dir = tmp_path / "models" / "Qwen3-1.7B"
    calls = []

    def fake_downloader(**kwargs):
        calls.append(kwargs)
        local_dir.mkdir(parents=True)
        write_minimal_checkpoint_dir(local_dir)
        return str(local_dir)

    resolved = resolve_model_dir(
        model="Qwen/Qwen3-1.7B",
        revision="abc123",
        download_dir=local_dir,
        downloader=fake_downloader,
    )

    assert resolved == str(local_dir)
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-1.7B",
            "revision": "abc123",
            "local_dir": str(local_dir),
        }
    ]


def test_model_resolver_uses_default_model_for_download(tmp_path) -> None:
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    write_minimal_checkpoint_dir(downloaded)

    resolved = resolve_model_dir(
        default_model="Qwen/Qwen3-1.7B",
        downloader=lambda **_: str(downloaded),
    )

    assert resolved == str(downloaded)


def test_model_resolver_rejects_invalid_local_model_dir(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing config.json"):
        resolve_model_dir(model_dir=tmp_path)


def test_model_resolver_missing_path_does_not_download() -> None:
    with pytest.raises(FileNotFoundError, match="model path does not exist"):
        resolve_model_dir(model="/does/not/exist", model_dir=None)

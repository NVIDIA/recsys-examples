import json


def write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_inspect_qwen3_checkpoint_builds_report(tmp_path) -> None:
    from inspect_qwen3_checkpoint import build_report

    config = {
        "_name_or_path": "Qwen/Qwen3-0.6B",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": False,
    }
    weight_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    write_json(tmp_path / "config.json", config)
    write_json(
        tmp_path / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                name: "model-00001-of-00001.safetensors"
                for name in weight_names
            },
        },
    )

    report = build_report(tmp_path, max_layers=1)

    assert report["model_name"] == "Qwen/Qwen3-0.6B"
    assert report["known_variant"] is None
    assert report["num_layers"] == 1
    assert report["num_load_requests"] > 0
    assert report["grouped_files"] == {"model-00001-of-00001.safetensors": 12}
    assert any(
        request["logical_name"] == "layers.0.self_attn.qkv_proj.weight"
        for request in report["sample_requests"]
    )


def test_inspect_qwen3_checkpoint_recognizes_1_7b_tied_embeddings(tmp_path) -> None:
    from inspect_qwen3_checkpoint import build_report

    config = {
        "_name_or_path": "Qwen/Qwen3-1.7B",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_hidden_layers": 28,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    weight_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    # Keep the manifest compact while still satisfying all 28 required layers.
    for layer_idx in range(1, 28):
        weight_names.extend(
            name.replace("model.layers.0", f"model.layers.{layer_idx}")
            for name in weight_names[2:11]
        )
    write_json(tmp_path / "config.json", config)
    write_json(
        tmp_path / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                name: "model-00001-of-00001.safetensors"
                for name in weight_names
            },
        },
    )

    report = build_report(tmp_path, max_layers=1)

    assert report["known_variant"] == "qwen3-1.7b"
    assert report["hidden_size"] == 2048
    assert report["q_size"] == 2048
    assert report["kv_size"] == 1024
    assert report["qkv_size"] == 4096
    assert report["gate_up_size"] == 12288
    assert report["tie_word_embeddings"] is True
    assert report["grouped_files"] == {"model-00001-of-00001.safetensors": 254}

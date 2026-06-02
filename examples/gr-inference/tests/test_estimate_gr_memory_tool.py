import json


def test_estimate_gr_memory_tool_emits_json(capsys) -> None:
    from estimate_gr_memory import main

    assert (
        main(
            [
                "--batch-size",
                "2",
                "--num-layers",
                "4",
                "--context-len",
                "10",
                "--max-decode-steps",
                "3",
                "--max-beam-width",
                "5",
                "--active-beam-width",
                "2",
                "--num-kv-heads",
                "2",
                "--head-dim",
                "8",
                "--vocab-size",
                "100",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["context_kv_bytes"] == 5120
    assert payload["beam_kv_bytes"] == 7680
    assert payload["logits_workspace_bytes"] == 1600


def test_estimate_gr_memory_tool_can_read_qwen3_config(tmp_path, capsys) -> None:
    from estimate_gr_memory import main

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "Qwen/Qwen3-1.7B",
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_hidden_layers": 28,
                "head_dim": 128,
                "vocab_size": 151936,
                "tie_word_embeddings": True,
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--model-dir",
                str(tmp_path),
                "--batch-size",
                "2",
                "--context-len",
                "10",
                "--max-decode-steps",
                "3",
                "--max-beam-width",
                "5",
                "--active-beam-width",
                "2",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["num_layers"] == 28
    assert payload["num_kv_heads"] == 8
    assert payload["head_dim"] == 128
    assert payload["vocab_size"] == 151936
    assert payload["context_kv_bytes"] == 2293760
    assert payload["logits_workspace_bytes"] == 2430976

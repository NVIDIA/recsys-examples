import argparse
import os

import numpy as np


def convert(ps_dir, dynamicemb_dir, key_filename, value_filename, emb_dim=None):
    keys_arr = np.fromfile(
        os.path.join(dynamicemb_dir, key_filename), dtype=np.int64
    ).reshape(-1)
    num_embeddings = keys_arr.shape[0]
    np.save(os.path.join(ps_dir, key_filename), keys_arr)

    values_arr = np.fromfile(
        os.path.join(dynamicemb_dir, value_filename), dtype=np.float32
    )
    assert (
        values_arr.shape[0] % num_embeddings == 0
    ), f"Wrong shape of values file {values_arr.shape}"
    values_arr = values_arr.reshape(num_embeddings, -1)
    if emb_dim is not None:
        assert (
            values_arr.shape[1] == emb_dim
        ), f"Wrong shape of values file {values_arr.shape}"
    np.save(os.path.join(ps_dir, value_filename), values_arr)


def convert_ckpt(checkpoint_dir, ps_module, dynamicemb_module, table_names, world_size):
    table_names = table_names.split(",")
    assert len(table_names) > 0, "The table names list is empty."

    dynamicemb_dir = os.path.join(checkpoint_dir, dynamicemb_module)
    ps_dir = os.path.join(checkpoint_dir, ps_module)
    os.makedirs(ps_dir, exist_ok=True)

    for name in table_names:
        print(f"Converting table[{name}] ...")

        for rank_id in range(world_size):
            key_filename = f"{name}_emb_keys.rank_{rank_id}.world_size_{world_size}"
            value_filename = f"{name}_emb_values.rank_{rank_id}.world_size_{world_size}"

            convert(ps_dir, dynamicemb_dir, key_filename, value_filename)

    print(f"Finished converting DynamicEmb checkpoint for Numpy format into {ps_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference End-to-end Example")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--ps_module", type=str, required=True)
    parser.add_argument("--dynamicemb_module", type=str, required=True)
    parser.add_argument("--table_names", type=str, required=True)
    parser.add_argument("--world_size", type=int, default=1)

    args = parser.parse_args()
    convert_ckpt(
        args.checkpoint_dir,
        args.ps_module,
        args.dynamicemb_module,
        args.table_names,
        args.world_size,
    )

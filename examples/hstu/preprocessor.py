# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys
import tarfile
from typing import Dict, List
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("main")

import time


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(
        "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
        % (percent, progress_size / (1024 * 1024), speed, duration)
    )
    sys.stdout.flush()


def _one_hot_encode(row):
    mapping = {category: i + 1 for i, category in enumerate(row.unique())}
    row = row.map(mapping)
    return row


class DataProcessor:
    """
    Abstract class for data processing.

    Args:
        download_url (str): URL from which to download the dataset.
        data_path (str): Path where the dataset will be stored.
        file_name (str): Name of the file containing the dataset.
        prefix (str): The root directory of the dataset.
    """

    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        self._download_url = download_url
        self._data_path = data_path
        os.makedirs(self._data_path, exist_ok=True)
        self._file_name = file_name
        self._prefix = prefix

    def _post_process(
        self,
        user_feature_df,
        sequence_feature_df,
        user_id_feature_name: str,
        contextual_feature_names: List[str],
        item_feature_name: str,
        action_feature_name: str,
        output_file,
    ) -> None:
        if user_feature_df is not None:
            final_df = pd.merge(
                sequence_feature_df, user_feature_df, on=user_id_feature_name
            )
        else:
            final_df = sequence_feature_df
        final_df.to_csv(output_file, index=False, sep=",")
        log.info(f"Processed file saved to {output_file}")
        log.info(f"num users: {len(final_df[user_id_feature_name])}")
        data = []
        for name in contextual_feature_names:
            data.append([name, final_df[name].min(), final_df[name].max()])
        log.info(["feature_name", "min", "max"])
        log.info(data)

        data = []
        for name in [item_feature_name, action_feature_name]:
            max_seq_len = int(final_df[name].apply(len).max())
            min_seq_len = int(final_df[name].apply(len).min())
            average_seq_len = int(final_df[name].apply(len).mean())
            min_id = int(final_df[name].apply(min).min())
            max_id = int(final_df[name].apply(max).max())
            data.append(
                [name, min_id, max_id, min_seq_len, max_seq_len, average_seq_len]
            )
        log.info(
            [
                "feature_name",
                "min",
                "max",
                "min_seqlen",
                "max_seqlen",
                "average_seqlen",
            ]
        )
        log.info(data)

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))


class MovielensDataProcessor(DataProcessor):
    """
    Data processor for the Movielens dataset.

    Args:
        download_url (str): URL from which to download the dataset.
        data_path (str): Path where the dataset will be stored.
        file_name (str): Name of the file containing the dataset.
        prefix (str): The root directory of the dataset.
    """

    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        super().__init__(download_url, data_path, file_name, prefix)
        self._item_feature_name = "movie_id"
        self._action_feature_name = "rating"
        if self._prefix == "ml-1m":
            self._contextual_feature_names = [
                "user_id",
                "sex",
                "age_group",
                "occupation",
                "zip_code",
            ]
            self._rating_mapping = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
            }
        else:
            assert self._prefix == "ml-20m"
            # ml-20m
            # ml-20m doesn't have user data.
            self._contextual_feature_names = [
                "user_id",
            ]
            self._rating_mapping = {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
                5: 4,
                6: 5,
                7: 6,
                8: 7,
                9: 8,
                10: 9,
            }
        self._output_file: str = os.path.join(data_path, prefix, "processed_seqs.csv")

    def download(self) -> None:
        """
        Download and decompress the dataset. The downloaded dataset will be saved in the "tmp" directory.
        """
        file_path = f"{self._data_path}{self._file_name}"
        if not self.file_exists(file_path):
            log.info(f"Downloading {self._download_url}")
            urlretrieve(self._download_url, file_path, reporthook)
        if file_path[-4:] == ".zip":
            ZipFile(file_path, "r").extractall(path=self._data_path)
        else:
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(self._data_path)

    def preprocess(self) -> None:
        """
        Preprocess the raw data. The support dataset are "ml-1m" and "ml-20m".
        """
        self.download()
        if self._prefix == "ml-1m":
            users = pd.read_csv(
                f"{self._data_path}{self._prefix}/users.dat",
                sep="::",
                names=self._contextual_feature_names,
            )
            log_df = pd.read_csv(
                f"{self._data_path}{self._prefix}/ratings.dat",
                sep="::",
                names=["user_id", "movie_id", "rating", "unix_timestamp"],
            )
        else:
            assert self._prefix == "ml-20m"
            # ml-20m
            # ml-20m doesn't have user data.
            users = None
            # ratings: userId,movieId,rating,timestamp
            log_df = pd.read_csv(
                f"{self._data_path}{self._prefix}/ratings.csv",
                sep=",",
            )
            log_df.rename(
                columns={
                    "userId": "user_id",
                    "movieId": "movie_id",
                    "timestamp": "unix_timestamp",
                },
                inplace=True,
            )
            log_df["rating"] = (log_df["rating"] * 2).astype(int)

        log_df["movie_id"] = log_df["movie_id"].astype(int)
        log_df["rating"] = log_df["rating"].map(self._rating_mapping).astype(int)
        df_grouped_by_user = log_df.groupby("user_id").agg(list).reset_index()

        contextual_feature_names = self._contextual_feature_names.copy()
        contextual_feature_names.remove("user_id")
        for col in contextual_feature_names:
            users[col] = _one_hot_encode(users[col])
        self._post_process(
            users,
            df_grouped_by_user,
            "user_id",
            contextual_feature_names=self._contextual_feature_names,
            item_feature_name=self._item_feature_name,
            action_feature_name=self._action_feature_name,
            output_file=self._output_file,
        )


class DLRMKuaiRandProcessor(DataProcessor):
    """

    Data processor for the `KuaiRand <https://kuairand.com/>`_ dataset.

    Args:
        download_url (str): URL from which to download the dataset.
        data_path (str): Path where the dataset will be stored.
        file_name (str): Name of the file containing the dataset.
        prefix (str): The root directory of the dataset.
    """

    def __init__(
        self,
        download_url: str,
        data_path: str,
        file_name: str,
        prefix: str,
    ) -> None:
        super().__init__(download_url, data_path, file_name, prefix)
        self._item_feature_name = "video_id"
        self._action_feature_name = "action_weights"
        self._contextual_feature_names = [
            "user_id",
            "user_active_degree",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
        ]
        base_path = os.path.join(data_path, prefix, "data")
        if prefix == "KuaiRand-Pure":
            self._log_files = [
                os.path.join(base_path, "log_standard_4_08_to_4_21_pure.csv"),
                os.path.join(base_path, "log_standard_4_22_to_5_08_pure.csv"),
            ]
            self._user_features_file = (
                os.path.join(base_path, "user_features_pure.csv")
            )

        elif prefix == "KuaiRand-1K":
            self._log_files = [
                os.path.join(base_path, "log_standard_4_08_to_4_21_1k.csv"),
                os.path.join(base_path, "log_standard_4_22_to_5_08_1k.csv"),
            ]
            self._user_features_file = os.path.join(base_path, "user_features_1k.csv")
        elif prefix == "KuaiRand-27K":
            self._log_files = [
                os.path.join(base_path, "log_standard_4_08_to_4_21_27k_part1.csv"),
                os.path.join(base_path, "log_standard_4_08_to_4_21_27k_part2.csv"),
                os.path.join(base_path, "log_standard_4_22_to_5_08_27k_part1.csv"),
                os.path.join(base_path, "log_standard_4_22_to_5_08_27k_part2.csv"),
            ]
            self._user_features_file = os.path.join(base_path, "user_features_27k.csv")
        self._output_file: str = os.path.join(base_path, "processed_seqs.csv")
        self._event_merge_weight: Dict[str, int] = {
            "is_click": 1,
            "is_like": 2,
            "is_follow": 4,
            "is_comment": 8,
            "is_forward": 16,
            "is_hate": 32,
            "long_view": 64,
            "is_profile_enter": 128,
        }

    def download(self) -> None:
        """
        Download and decompress the dataset. The downloaded dataset will be saved in the "tmp" directory.
        """
        file_path = f"{self._data_path}{self._file_name}"
        if not self.file_exists(file_path):
            log.info(f"Downloading {self._download_url}")
            urlretrieve(self._download_url, file_path, reporthook)
            log.info(f"Downloaded to {file_path}")
            with tarfile.open(file_path, "r:*") as tar_ref:
                tar_ref.extractall(path=self._data_path)
                log.info("Data files extracted")

    def preprocess(self) -> None:
        """
        Preprocess the raw data. The support dataset are "KuaiRand-Pure", "KuaiRand-1K", "KuaiRand-27K".
        """
        self.download()
        log.info("Preprocessing data...")
        seq_cols = [
            "video_id",
            "time_ms",
            "action_weights",
            "play_time_ms",
            "duration_ms",
        ]
        df = None
        for idx, log_file in enumerate(self._log_files):
            log.info(f"Processing {log_file}...")
            log_df = pd.read_csv(
                log_file,
                delimiter=",",
            )
            df_grouped_by_user = log_df.groupby("user_id").agg(list).reset_index()

            for event, weight in self._event_merge_weight.items():
                df_grouped_by_user[event] = df_grouped_by_user[event].apply(
                    lambda seq: np.where(np.array(seq) == 0, 0, weight)
                )

            events = list(self._event_merge_weight.keys())
            df_grouped_by_user["action_weights"] = df_grouped_by_user.apply(
                lambda row: [int(sum(x)) for x in zip(*[row[col] for col in events])],
                axis=1,
            )
            df_grouped_by_user = df_grouped_by_user[["user_id"] + seq_cols]

            if idx == 0:
                df = df_grouped_by_user
            else:
                df = df.merge(df_grouped_by_user, on="user_id", suffixes=("_x", "_y"))  # type: ignore[union-attr]
                for col in seq_cols:
                    df[col] = df.apply(
                        lambda row: row[col + "_x"] + row[col + "_y"], axis=1
                    )
                    df = df.drop(columns=[col + "_x", col + "_y"])

        log.info("Merging user features...")
        user_features_df = pd.read_csv(self._user_features_file, delimiter=",")

        contextual_feature_names = self._contextual_feature_names.copy()
        contextual_feature_names.remove("user_id")
        for col in contextual_feature_names:
            user_features_df[col] = _one_hot_encode(user_features_df[col])

        self._post_process(
            user_features_df,
            df,
            "user_id",
            contextual_feature_names=self._contextual_feature_names,
            item_feature_name="video_id",
            action_feature_name="action_weights",
            output_file=self._output_file,
        )


dataset_names = (
    "ml-1m",
    "ml-20m",
    "kuairand-pure",
    "kuairand-1k",
    "kuairand-27k",
)

def get_common_preprocessors(dataset_path: str):
    """
    Get common data preprocessors.

    Returns:
        dict: Dictionary of common data preprocessors. The valid keys are
        "ml-1m", "ml-20m", "kuairand-pure", "kuairand-1k", "kuairand-27k".
    """
    data_path = dataset_path if dataset_path else "tmp_data"
    ml_1m_dp = MovielensDataProcessor(
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        data_path=data_path,
        file_name="movielens1m.zip",
        prefix="ml-1m",
    )
    ml_20m_dp = MovielensDataProcessor(
        "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        data_path=data_path,
        file_name="movielens20m.zip",
        prefix="ml-20m",
    )
    kuairand_pure_dp = DLRMKuaiRandProcessor(
        download_url="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz",
        data_path=data_path,
        file_name="KuaiRand-Pure.tar.gz",
        prefix="KuaiRand-Pure",
    )
    kuairand_1k_dp = DLRMKuaiRandProcessor(
        download_url="https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz",
        data_path=data_path,
        file_name="KuaiRand-1K.tar.gz",
        prefix="KuaiRand-1K",
    )
    kuairand_27k_dp = DLRMKuaiRandProcessor(
        download_url="https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz",
        data_path=data_path,
        file_name="KuaiRand-27K.tar.gz",
        prefix="KuaiRand-27K",
    )
    return {key: locals()[f"{key}_dp".replace('-', '_')] for key in dataset_names}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessor")
    parser.add_argument(
        "--dataset_name", choices=list(dataset_names) + ["all"]
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    common_preprocessors = get_common_preprocessors(args.dataset_path)
    if args.dataset_name == "all":
        for dataset_name in common_preprocessors.keys():
            dp = common_preprocessors[dataset_name]
            dp.preprocess()
    else:
        dp = common_preprocessors[args.dataset_name]
        dp.preprocess()

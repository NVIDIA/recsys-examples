# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import socket
import stat
import struct
import tempfile
import threading
import time
from array import array
from ctypes import CDLL, get_errno
from typing import List, Optional


DEFAULT_LAYERWISE_EVENTFD_SOCKET = "/tmp/flexkv_layerwise_eventfd.sock"


def create_layerwise_eventfd_socket_path() -> str:
    socket_directory = tempfile.mkdtemp(
        prefix=f"recsys-flexkv-layerwise-{os.geteuid()}-"
    )
    os.chmod(socket_directory, 0o700)
    return os.path.join(socket_directory, "eventfd.sock")


class FlexKVLayerwiseEventfdSender:
    """Creates layerwise eventfds and sends them to FlexKV's worker."""

    def __init__(
        self,
        num_layers: int,
        socket_path: str,
        num_counters: int = 3,
        timeout_s: float = 180.0,
    ) -> None:
        self.num_layers = int(num_layers)
        self.socket_path = socket_path
        self.num_counters = int(num_counters)
        self.timeout_s = float(timeout_s)
        self._eventfds: Optional[List[List[int]]] = None
        self._thread: Optional[threading.Thread] = None
        self._handoff_done = threading.Event()
        self._handoff_error: Optional[BaseException] = None

    @staticmethod
    def _create_eventfd() -> int:
        if hasattr(os, "eventfd"):
            return os.eventfd(0, 0)
        libc = CDLL(None, use_errno=True)
        fd = libc.eventfd(0, 0)
        if fd < 0:
            raise OSError(get_errno(), "eventfd creation failed")
        return int(fd)

    def create_eventfds(self) -> List[List[int]]:
        if self._eventfds is None:
            # FlexKV layerwise worker expects counter sets for triple buffering.
            self._eventfds = [
                [self._create_eventfd() for _ in range(self.num_layers)]
                for _ in range(self.num_counters)
            ]
        return self._eventfds

    def layer_eventfds(self, counter_id: int) -> List[int]:
        eventfds = self.create_eventfds()
        if counter_id < 0 or counter_id >= len(eventfds):
            raise ValueError(
                f"Invalid layerwise counter_id={counter_id}, "
                f"expected [0, {len(eventfds)})"
            )
        return eventfds[counter_id]

    def start(self) -> None:
        if self._thread is not None:
            return
        self.create_eventfds()
        self._thread = threading.Thread(
            target=self._run_sender,
            name="flexkv-layerwise-eventfd-sender",
            daemon=True,
        )
        self._thread.start()

    def wait_until_ready(self, timeout_s: Optional[float] = None) -> None:
        timeout = self.timeout_s + 1.0 if timeout_s is None else float(timeout_s)
        if not self._handoff_done.wait(timeout):
            raise TimeoutError(
                "Timed out waiting for FlexKV layerwise eventfd handoff "
                f"on socket {self.socket_path}"
            )
        if self._handoff_error is not None:
            raise RuntimeError(
                "FlexKV layerwise eventfd handoff failed"
            ) from self._handoff_error

    def _run_sender(self) -> None:
        try:
            self._send_eventfds()
        except BaseException as error:
            self._handoff_error = error
        finally:
            self._handoff_done.set()

    @staticmethod
    def _is_process_descendant(process_id: int, ancestor_id: int) -> bool:
        current_id = int(process_id)
        ancestor_id = int(ancestor_id)
        visited = set()
        while current_id > 1 and current_id not in visited:
            if current_id == ancestor_id:
                return True
            visited.add(current_id)
            try:
                with open(f"/proc/{current_id}/stat", encoding="utf-8") as stat_file:
                    process_stat = stat_file.read()
                fields_after_name = process_stat[process_stat.rfind(")") + 1 :].split()
                current_id = int(fields_after_name[1])
            except (OSError, IndexError, ValueError):
                return False
        return current_id == ancestor_id

    def _authenticate_peer(self, sock: socket.socket) -> None:
        socket_stat = os.stat(self.socket_path, follow_symlinks=False)
        if not stat.S_ISSOCK(socket_stat.st_mode):
            raise PermissionError(
                f"Layerwise eventfd path is not a socket: {self.socket_path}"
            )
        if socket_stat.st_uid != os.geteuid():
            raise PermissionError(
                "FlexKV layerwise socket owner does not match the current user"
            )

        if not hasattr(socket, "SO_PEERCRED"):
            raise RuntimeError("SO_PEERCRED is required for layerwise eventfd handoff")
        credentials_size = struct.calcsize("3i")
        peer_pid, peer_uid, peer_gid = struct.unpack(
            "3i",
            sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, credentials_size),
        )
        if peer_uid != os.geteuid() or peer_gid != os.getegid():
            raise PermissionError(
                "FlexKV layerwise peer credentials do not match the current process"
            )
        if not self._is_process_descendant(peer_pid, os.getpid()):
            raise PermissionError(
                f"FlexKV layerwise peer pid {peer_pid} is outside the current process tree"
            )

    def _send_eventfds(self) -> None:
        eventfds = self.create_eventfds()
        deadline = time.time() + self.timeout_s
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.settimeout(min(1.0, max(deadline - time.time(), 0.01)))
                sock.connect(self.socket_path)
                self._authenticate_peer(sock)
                metadata = struct.pack(
                    "iiii",
                    0,
                    1,
                    self.num_layers,
                    len(eventfds),
                )
                sock.sendall(metadata)
                for counter_id, fds in enumerate(eventfds):
                    fd_array = array("i", fds)
                    sock.sendmsg(
                        [struct.pack("i", counter_id)],
                        [
                            (
                                socket.SOL_SOCKET,
                                socket.SCM_RIGHTS,
                                fd_array.tobytes(),
                            )
                        ],
                    )
                ack = sock.recv(1)
                if ack != b"\x01":
                    raise RuntimeError(
                        f"FlexKV layerwise eventfd receiver returned ack={ack!r}"
                    )
                return
            except (OSError, RuntimeError) as e:
                last_error = e
                time.sleep(0.05)
            finally:
                sock.close()
        raise RuntimeError(
            "Timed out sending layerwise eventfds to FlexKV "
            f"socket {self.socket_path}: {last_error}"
        )

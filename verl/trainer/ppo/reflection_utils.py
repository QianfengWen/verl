# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for selecting reflection candidates."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _as_numpy(array: object) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _mix_seed(seed: int, step: int) -> int:
    mixed = (np.uint64(seed) + np.uint64(step) * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return int(mixed)


def _build_response_keys(responses: np.ndarray, response_mask: Optional[np.ndarray]) -> list[tuple[str, tuple[int, ...], bytes]]:
    keys: list[tuple[str, tuple[int, ...], bytes]] = []
    if response_mask is not None:
        response_mask = response_mask.astype(bool)
    for i in range(len(responses)):
        tokens = responses[i]
        if not isinstance(tokens, np.ndarray):
            tokens = np.asarray(tokens)
        if response_mask is not None:
            tokens = tokens[response_mask[i]]
        if tokens.dtype == object:
            tokens = np.asarray(tokens.tolist())
        keys.append((str(tokens.dtype), tokens.shape, tokens.tobytes()))
    return keys


def select_reflection_indices(
    uids: np.ndarray | list,
    responses: np.ndarray | torch.Tensor | list,
    is_correct: np.ndarray | list,
    *,
    num_select: int,
    response_mask: Optional[np.ndarray | torch.Tensor | list] = None,
    seed: int = 0,
    step: int = 0,
) -> list[int]:
    """Select indices for reflection with wrong-first, unique-response sampling.

    The selection prioritizes wrong responses, first choosing at most one wrong
    response per unique uid, then filling with remaining wrong responses, and
    finally with correct responses if needed. Duplicate responses (after applying
    response_mask) are never selected.
    """

    responses_np = _as_numpy(responses)
    uids_np = _as_numpy(uids).reshape(-1)
    is_correct_np = _as_numpy(is_correct).astype(bool).reshape(-1)
    if response_mask is not None:
        response_mask_np = _as_numpy(response_mask)
    else:
        response_mask_np = None

    total = len(responses_np)
    if num_select <= 0 or total == 0:
        return []
    if len(uids_np) != total or len(is_correct_np) != total:
        raise ValueError("uids, responses, and is_correct must have matching lengths.")

    response_keys = _build_response_keys(responses_np, response_mask_np)
    rng = np.random.default_rng(_mix_seed(int(seed), int(step)))

    indices = np.arange(total)
    wrong_indices = indices[~is_correct_np]
    correct_indices = indices[is_correct_np]

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_keys: set[tuple[str, tuple[int, ...], bytes]] = set()

    uid_to_wrong: dict[object, list[int]] = {}
    for idx in wrong_indices.tolist():
        uid_to_wrong.setdefault(uids_np[idx], []).append(int(idx))

    uid_list = list(uid_to_wrong.keys())
    if len(uid_list) > 1:
        uid_list = rng.permutation(uid_list).tolist()

    for uid in uid_list:
        if len(selected) >= num_select:
            break
        candidates = uid_to_wrong[uid]
        if len(candidates) > 1:
            candidates = rng.permutation(candidates).tolist()
        for idx in candidates:
            key = response_keys[idx]
            if key in selected_keys:
                continue
            selected.append(int(idx))
            selected_set.add(int(idx))
            selected_keys.add(key)
            break

    if len(selected) < num_select and len(wrong_indices) > 0:
        remaining_wrong = [
            int(idx)
            for idx in wrong_indices.tolist()
            if int(idx) not in selected_set and response_keys[int(idx)] not in selected_keys
        ]
        if remaining_wrong:
            remaining_wrong = rng.permutation(remaining_wrong).tolist()
            for idx in remaining_wrong:
                if len(selected) >= num_select:
                    break
                key = response_keys[idx]
                if key in selected_keys:
                    continue
                selected.append(int(idx))
                selected_set.add(int(idx))
                selected_keys.add(key)

    if len(selected) < num_select and len(correct_indices) > 0:
        remaining_correct = [
            int(idx)
            for idx in correct_indices.tolist()
            if int(idx) not in selected_set and response_keys[int(idx)] not in selected_keys
        ]
        if remaining_correct:
            remaining_correct = rng.permutation(remaining_correct).tolist()
            for idx in remaining_correct:
                if len(selected) >= num_select:
                    break
                key = response_keys[idx]
                if key in selected_keys:
                    continue
                selected.append(int(idx))
                selected_set.add(int(idx))
                selected_keys.add(key)

    return selected

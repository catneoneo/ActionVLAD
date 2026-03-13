# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
"""Utilities for loading pretrained PyTorch checkpoints into another model.

This module is meant for transfer-learning scenarios where:
1) checkpoint and target model names are not exactly the same, and/or
2) output head shapes differ.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import torch


LoadReport = collections.namedtuple(
    'LoadReport',
    [
        'loaded',
        'shape_mismatch',
        'missing_in_checkpoint',
        'unused_in_checkpoint',
    ],
)


def _normalize_checkpoint_state(checkpoint):
    """Extract a state_dict from common checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'model', 'model_state_dict', 'net'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError('Unsupported checkpoint format: %s' % type(checkpoint))


def _strip_prefix(name, prefixes):
    for prefix in prefixes:
        if prefix and name.startswith(prefix):
            return name[len(prefix):]
    return name


def load_pretrained_to_model(
    target_model,
    checkpoint_path,
    map_location='cpu',
    source_prefix_to_strip=('module.',),
    target_prefix_to_add='',
    include_keywords=None,
    exclude_keywords=None,
    strict=False,
    verbose=True,
):
    """Load pretrained weights into ``target_model`` with flexible mapping.

    Args:
        target_model: torch.nn.Module to receive pretrained weights.
        checkpoint_path: Path to checkpoint file.
        map_location: torch.load map_location.
        source_prefix_to_strip: Prefix(es) to strip from checkpoint names.
        target_prefix_to_add: Prefix to prepend before matching target names.
        include_keywords: Optional iterable; only params containing any keyword are loaded.
        exclude_keywords: Optional iterable; params containing any keyword are skipped.
        strict: Passed to model.load_state_dict.
        verbose: Print summary to stdout.

    Returns:
        LoadReport: details about matched/skipped parameters.
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    source_state = _normalize_checkpoint_state(checkpoint)
    target_state = target_model.state_dict()

    include_keywords = tuple(include_keywords or ())
    exclude_keywords = tuple(exclude_keywords or ())

    filtered_state = {}
    loaded = []
    shape_mismatch = []
    unused_in_checkpoint = []

    for source_name, source_tensor in source_state.items():
        mapped_name = _strip_prefix(source_name, source_prefix_to_strip)
        mapped_name = '%s%s' % (target_prefix_to_add, mapped_name)

        if include_keywords and not any(k in mapped_name for k in include_keywords):
            unused_in_checkpoint.append(source_name)
            continue
        if exclude_keywords and any(k in mapped_name for k in exclude_keywords):
            unused_in_checkpoint.append(source_name)
            continue

        if mapped_name not in target_state:
            unused_in_checkpoint.append(source_name)
            continue

        if tuple(source_tensor.shape) != tuple(target_state[mapped_name].shape):
            shape_mismatch.append(
                (source_name, mapped_name, tuple(source_tensor.shape), tuple(target_state[mapped_name].shape))
            )
            continue

        filtered_state[mapped_name] = source_tensor
        loaded.append((source_name, mapped_name))

    missing_in_checkpoint = [name for name in target_state.keys() if name not in filtered_state]

    target_model.load_state_dict(filtered_state, strict=strict)

    report = LoadReport(
        loaded=loaded,
        shape_mismatch=shape_mismatch,
        missing_in_checkpoint=missing_in_checkpoint,
        unused_in_checkpoint=unused_in_checkpoint,
    )

    if verbose:
        print('Loaded params: %d' % len(report.loaded))
        print('Shape mismatch: %d' % len(report.shape_mismatch))
        print('Missing in checkpoint (not loaded): %d' % len(report.missing_in_checkpoint))
        print('Unused checkpoint params: %d' % len(report.unused_in_checkpoint))

    return report

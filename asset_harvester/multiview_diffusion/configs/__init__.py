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

"""
Unified SparseViewDiT configuration: model architecture + training defaults.

``sparseviewdit_512.yaml`` in this package is the checked-in source of truth
for the released checkpoint. ``load_unified_config`` reads it (or a
user-supplied override file) and validates its structure.
"""

from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent / "sparseviewdit_512.yaml"

_REQUIRED_SECTIONS = ("model", "training")


def load_unified_config(path: str | Path | None = None) -> dict:
    """Load a unified config YAML with ``model`` and ``training`` sections.

    ``path=None`` loads the packaged default (the released 512px checkpoint's
    configuration).
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping with sections {_REQUIRED_SECTIONS}: {config_path}")
    for section in _REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Config is missing the '{section}' section: {config_path}")
    return config

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

"""Tests for the unified model/training config (source of truth YAML)."""

import inspect

import pytest
import yaml

from asset_harvester.multiview_diffusion.configs import DEFAULT_CONFIG_PATH, load_unified_config
from asset_harvester.multiview_diffusion.models.sparseviewdit import SparseViewDiTTransformer2DModelNative
from asset_harvester.multiview_diffusion.training.train import parse_args

_MINIMAL_CLI = ["--data_root", "x", "--checkpoint_path", "y"]


def test_default_config_loads_and_has_sections():
    config = load_unified_config()
    assert set(config) >= {"model", "training"}
    assert config["model"]["num_attention_heads"] * config["model"]["attention_head_dim"] == 2240


def test_model_section_matches_transformer_signature():
    # Every model key must be an actual constructor parameter, so the YAML
    # cannot drift from the model class.
    config = load_unified_config()
    params = set(inspect.signature(SparseViewDiTTransformer2DModelNative.__init__).parameters)
    unknown = set(config["model"]) - params
    assert not unknown, f"model config keys not accepted by the transformer: {unknown}"


def test_model_builder_uses_unified_config():
    from asset_harvester.multiview_diffusion.utils.model_builder import _TRANSFORMER_CONFIG

    assert _TRANSFORMER_CONFIG == load_unified_config()["model"]


def test_parse_args_defaults_come_from_yaml():
    config = load_unified_config()
    args = parse_args(_MINIMAL_CLI)
    for key, expected in config["training"].items():
        if key == "lora_alpha":  # resolved to lora_rank post-parse
            expected = config["training"]["lora_rank"]
        assert getattr(args, key) == expected, f"{key}: {getattr(args, key)} != {expected}"
    assert args.model_config == config["model"]


def test_cli_overrides_yaml_defaults():
    args = parse_args(_MINIMAL_CLI + ["--learning_rate", "3e-4", "--use_ema", "--flow_shift", "3.0"])
    assert args.learning_rate == pytest.approx(3e-4)
    assert args.use_ema is True
    assert args.flow_shift == pytest.approx(3.0)
    # Boolean flags are negatable so a YAML `true` can be turned off on the CLI.
    args = parse_args(_MINIMAL_CLI + ["--no-use_ema"])
    assert args.use_ema is False


def test_custom_config_file_overrides_defaults(tmp_path):
    config = load_unified_config()
    config["training"]["learning_rate"] = 7e-6
    config["training"]["max_train_steps"] = 123
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(config))

    args = parse_args(["--config", str(path)] + _MINIMAL_CLI)
    assert args.learning_rate == pytest.approx(7e-6)
    assert args.max_train_steps == 123
    # CLI still wins over the custom file.
    args = parse_args(["--config", str(path)] + _MINIMAL_CLI + ["--max_train_steps", "5"])
    assert args.max_train_steps == 5


def test_unknown_training_key_rejected(tmp_path):
    config = load_unified_config()
    config["training"]["not_a_real_option"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(path)] + _MINIMAL_CLI)


def test_missing_section_rejected(tmp_path):
    path = tmp_path / "no_model.yaml"
    path.write_text(yaml.safe_dump({"training": {}}))
    with pytest.raises(ValueError, match="model"):
        load_unified_config(path)


def test_default_path_is_packaged():
    assert DEFAULT_CONFIG_PATH.name == "sparseviewdit_512.yaml"
    assert DEFAULT_CONFIG_PATH.is_file()

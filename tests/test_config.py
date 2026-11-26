"""Unit tests for configuration validation."""

import pytest

from src.config.loader import ConfigError, validate_export, validate_split, validate_training


def test_valid_split() -> None:
    validate_split({"train": 0.8, "validation": 0.1, "test": 0.1})


def test_split_must_sum_to_one() -> None:
    with pytest.raises(ConfigError):
        validate_split({"train": 0.8, "validation": 0.1})


def test_export_requires_paths() -> None:
    with pytest.raises(ConfigError):
        validate_export({})


def test_training_requires_positive_batch_settings() -> None:
    with pytest.raises(ConfigError):
        validate_training({"model": "m", "dataset_config": "d", "dataset_group": "g",
                           "output_dir": "o", "per_device_train_batch_size": 0}, "sft")

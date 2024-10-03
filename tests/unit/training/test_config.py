from typing import Optional

import pytest

from sae_lens import __version__
from sae_lens.config import CacheActivationsRunnerConfig, LanguageModelSAERunnerConfig

TINYSTORIES_MODEL = "tiny-stories-1M"
TINYSTORIES_DATASET = "roneneldan/TinyStories"


def test_sae_training_runner_config_runs_with_defaults():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    _ = LanguageModelSAERunnerConfig()

    assert True


def test_sae_training_runner_config_total_training_tokens():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    cfg = LanguageModelSAERunnerConfig()

    assert cfg.total_training_tokens == 2000000


def test_sae_training_runner_config_total_training_steps():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    cfg = LanguageModelSAERunnerConfig()

    assert cfg.total_training_steps == 488


def test_sae_training_runner_config_get_sae_base_parameters():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    cfg = LanguageModelSAERunnerConfig()

    expected_config = {
        "architecture": "standard",
        "d_in": 512,
        "d_sae": 2048,
        "activation_fn_str": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": True,
        "dtype": "float32",
        "model_name": "gelu-2l",
        "hook_name": "blocks.0.hook_mlp_out",
        "hook_layer": 0,
        "hook_head_index": None,
        "device": "cpu",
        "context_size": 128,
        "start_pos_offset": 0,
        "end_pos_offset": 0,
        "prepend_bos": True,
        "finetuning_scaling_factor": False,
        "dataset_path": "",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": str(__version__),
        "normalize_activations": "none",
        "model_from_pretrained_kwargs": {
            "center_writing_weights": False,
        },
    }
    assert expected_config == cfg.get_base_sae_cfg_dict()


def test_sae_training_runner_config_raises_error_if_resume_true():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    with pytest.raises(ValueError):
        _ = LanguageModelSAERunnerConfig(resume=True)
    assert True


def test_sae_training_runner_config_raises_error_if_d_sae_and_expansion_factor_not_none():
    with pytest.raises(ValueError):
        _ = LanguageModelSAERunnerConfig(d_sae=128, expansion_factor=4)
    assert True


def test_sae_training_runner_config_expansion_factor():
    cfg = LanguageModelSAERunnerConfig()

    assert cfg.expansion_factor == 4


@pytest.mark.parametrize(
    "start_pos_offset, end_pos_offset, expected_error",
    [
        (-1, 0, ValueError),
        (0, 0, None),
        (10, 0, None),
        (11, 0, ValueError),
        (0, -1, ValueError),
        (0, 10, ValueError),
        (0, 11, ValueError),
        (5, 5, None),
        (6, 5, ValueError),
        (3, 4, None),
    ],
)
def test_sae_training_runner_config_start_end_pos_offset(
    start_pos_offset: int, end_pos_offset: int, expected_error: Optional[ValueError]
):
    context_size = 10
    if expected_error is ValueError:
        with pytest.raises(expected_error):
            LanguageModelSAERunnerConfig(
                start_pos_offset=start_pos_offset,
                end_pos_offset=end_pos_offset,
                context_size=context_size,
            )
    else:
        LanguageModelSAERunnerConfig(
            start_pos_offset=start_pos_offset,
            end_pos_offset=end_pos_offset,
            context_size=context_size,
        )


@pytest.mark.parametrize(
    "start_pos_offset, end_pos_offset, expected_error",
    [
        (-1, 0, ValueError),
        (0, 0, None),
        (10, 0, None),
        (11, 0, ValueError),
        (0, -1, ValueError),
        (0, 10, ValueError),
        (0, 11, ValueError),
        (5, 5, None),
        (6, 5, ValueError),
        (3, 4, None),
    ],
)
def test_cache_activations_runner_config_start_end_pos_offset(
    start_pos_offset: int, end_pos_offset: int, expected_error: Optional[ValueError]
):
    context_size = 10
    if expected_error is ValueError:
        with pytest.raises(expected_error):
            CacheActivationsRunnerConfig(
                start_pos_offset=start_pos_offset,
                end_pos_offset=end_pos_offset,
                context_size=context_size,
            )
    else:
        CacheActivationsRunnerConfig(
            start_pos_offset=start_pos_offset,
            end_pos_offset=end_pos_offset,
            context_size=context_size,
        )

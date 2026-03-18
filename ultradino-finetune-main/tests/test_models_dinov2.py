from pathlib import Path


import pytest
import torch


from ultradino_finetune.models import load_pretrained_ultradino, load_from_scratch
from ultradino_finetune.models.dinov2.load import get_param_groups_with_decay


TEST_DATA = {
    "vits16": {
        "relpath": Path("pretrained/vits16_dinus_13m+20241205-132822-468746.pth"),
    },
    "vitb16": {
        "relpath": Path("pretrained/vitb16_dinus_13m+20250115-101016-411691.pth"),
    },
}


@pytest.mark.parametrize("model_type", TEST_DATA.keys())
def test_load_pretrained_ultradino(test_data_base_path, model_type):
    test_data = TEST_DATA[model_type]
    weights_path = test_data_base_path / test_data["relpath"]

    assert weights_path.is_file()

    model = load_pretrained_ultradino(model_type, weights_path)


def test_load_from_scratch_and_get_param_groups():
    """Test loading vits16 from scratch and getting param groups with decay."""
    model_type = "vits16"

    # Load model from scratch
    model = load_from_scratch(model_type, device="cpu")

    # Verify model is created correctly
    assert model is not None
    assert hasattr(model, "parameters")

    # Count total parameters to ensure model is properly initialized
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model should have parameters"

    # Get param groups with decay
    param_groups = get_param_groups_with_decay(
        model, lr_decay_rate=0.65, patch_embed_lr_mult=0.2
    )

    # Verify param groups structure
    assert hasattr(
        param_groups, "__iter__"
    ), "Should return an iterable of param groups"
    param_groups_list = list(param_groups)
    assert len(param_groups_list) > 0, "Should have at least one param group"

    # Verify each param group has required keys
    for group in param_groups_list:
        assert isinstance(group, dict), "Each param group should be a dict"
        assert "params" in group, "Each param group should have 'params' key"
        assert "foreach" in group, "Each param group should have 'foreach' key"
        assert group["foreach"] is True, "foreach should be set to True"

        # Verify params are torch tensors
        assert isinstance(group["params"], list), "params should be a list"
        for param in group["params"]:
            assert isinstance(
                param, torch.Tensor
            ), "Each param should be a torch.Tensor"

    # Count total parameters in param groups should match model parameters
    total_params_in_groups = sum(
        sum(p.numel() for p in group["params"]) for group in param_groups_list
    )
    assert (
        total_params_in_groups == total_params
    ), "Total parameters in groups should match model parameters"


def test_optimizer_with_learning_rate_decay():
    """Test optimizer applies lr decay correctly via param groups."""
    model_type = "vits16"
    base_lr = 1e-3
    lr_decay_rate = 0.65
    patch_embed_lr_mult = 0.2

    # Load model from scratch
    model = load_from_scratch(model_type, device="cpu")

    # Get param groups with decay
    param_groups = get_param_groups_with_decay(
        model,
        lr_decay_rate=lr_decay_rate,
        patch_embed_lr_mult=patch_embed_lr_mult,
    )

    param_groups_list = list(param_groups)

    # Create optimizer param groups with base learning rate
    optimizer_param_groups = []
    for group in param_groups_list:
        optimizer_group = {
            "params": group["params"],
            "lr": base_lr * group["lr_multiplier"],
            "weight_decay": 0.05 * group["wd_multiplier"],
        }
        optimizer_param_groups.append(optimizer_group)

    # Create optimizer
    optimizer = torch.optim.AdamW(optimizer_param_groups)

    # Verify optimizer param groups have correct learning rates
    assert len(optimizer.param_groups) == len(param_groups_list)

    # Track expected vs actual learning rates
    lr_multipliers_found = []
    for i, (orig_group, opt_group) in enumerate(
        zip(param_groups_list, optimizer.param_groups)
    ):
        expected_lr = base_lr * orig_group["lr_multiplier"]
        actual_lr = opt_group["lr"]

        assert abs(expected_lr - actual_lr) < 1e-10, (
            f"Group {i}: Expected lr {expected_lr}, got {actual_lr}"
        )

        # Verify weight decay is also applied correctly
        expected_wd = 0.05 * orig_group["wd_multiplier"]
        actual_wd = opt_group["weight_decay"]

        assert abs(expected_wd - actual_wd) < 1e-10, (
            f"Group {i}: Expected weight_decay {expected_wd}, got {actual_wd}"
        )

        lr_multipliers_found.append(orig_group["lr_multiplier"])

    # Verify we have different learning rate multipliers
    unique_multipliers = set(lr_multipliers_found)
    assert len(unique_multipliers) > 1, (
        "Should have multiple different lr_multipliers for layer-wise decay"
    )

    # Verify we have the expected range of multipliers
    min_multiplier = min(lr_multipliers_found)
    max_multiplier = max(lr_multipliers_found)

    # For vits16 with 12 layers and decay rate 0.65, we expect:
    # - Some layers to have full learning rate (1.0)
    # - Some layers to have decayed learning rates
    # - Patch embedding layers to have reduced learning rate
    assert max_multiplier <= 1.0, "Max multiplier should not exceed 1.0"
    assert min_multiplier > 0.0, "Min multiplier should be positive"

    # Check that we have patch embed multiplier effect
    patch_embed_multipliers = [
        mult for mult in lr_multipliers_found
        if abs(mult - patch_embed_lr_mult) < 0.1
        or abs(mult - patch_embed_lr_mult * lr_decay_rate) < 0.1
    ]
    assert len(patch_embed_multipliers) > 0, (
        "Should have parameters with patch embedding learning rate multiplier"
    )

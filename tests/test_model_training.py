
import torch
from trashsorting.model import TrashModel

def test_training_step_runs():
    model = TrashModel(freeze_backbone=True)
    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, model.num_classes, (4,))
    batch = (x, y)

    loss = model.training_step(batch, 0)
    assert loss is not None
    assert loss.dim() == 0  # scalar loss

def test_only_classifier_gradients_change():
    model = TrashModel(freeze_backbone=True)

    x = torch.randn(4, 3, 224, 224)
    y = torch.randint(0, model.num_classes, (4,))
    loss = model.training_step((x, y), 0)
    loss.backward()

    # Backbone: NO gradients
    for name, p in model.baseline_model.named_parameters():
        if "classifier" not in name.lower():
            assert p.grad is None

    # Classifier: HAS gradients
    for p in model.baseline_model.get_classifier().parameters():
        assert p.grad is not None

def test_optimizer_has_correct_params():
    model = TrashModel(freeze_backbone=True)
    optim = model.configure_optimizers()
    for group in optim.param_groups:
        for p in group['params']:
            # The optimizer only contains parameters that are trainable
            assert p.requires_grad is True

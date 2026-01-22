
import torch
from trashsorting.model import TrashModel

def test_model_construction():
    # Check TrashModel is instantiated correctly
    model = TrashModel(freeze_backbone=True)
    assert model is not None

def test_freeze_backbone_true():
    model = TrashModel(freeze_backbone=True)
    # Backbone params frozen
    for name, p in model.baseline_model.named_parameters():
        if "classifier" not in name.lower():
            assert p.requires_grad is False

def test_freeze_backbone_false():
    model = TrashModel(freeze_backbone=False)
    # All params trainable
    for p in model.baseline_model.parameters():
        assert p.requires_grad is True

def test_classifier_head_trainable():
    model = TrashModel(freeze_backbone=True)
    for p in model.baseline_model.get_classifier().parameters():
        assert p.requires_grad is True

def test_batchnorm_eval_when_frozen():
    model = TrashModel(freeze_backbone=True)
    model.on_train_epoch_start()
    for m in model.baseline_model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            assert m.training is False
            assert m.track_running_stats is False

def test_forward_shape():
    model = TrashModel(freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, model.num_classes)

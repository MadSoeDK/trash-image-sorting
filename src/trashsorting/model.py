from torch import nn
import torch
from pytorch_lightning import LightningModule
import timm
import logging

logger = logging.getLogger(__name__)

class TrashModel(LightningModule):
    def __init__(
        self, 
        model_name="mobilenetv3_small_100", 
        num_classes: int = 6, 
        lr: float = 1e-3, 
        pretrained: bool = True,
        freeze_backbone: bool = True # NOT unused! Collected by self.save_hyperparameters() below.
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr: float = lr
        self.model_name: str = model_name
        self.num_classes: int = num_classes

        # Create model from timm
        self.baseline_model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

        # Freeze backbone immediately upon initializing TrashModel (important for unit tests + consistency)
        if freeze_backbone:
            self.freeze_backbone_keep_head()
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.baseline_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        if self._trainer is not None:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        if self._trainer is not None:
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        if self._trainer is not None:
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.baseline_model.parameters())
        return torch.optim.Adam(params, lr=self.lr)

    def freeze_backbone_keep_head(self):
        # freeze everything
        for p in self.baseline_model.parameters():
            p.requires_grad = False
        # unfreeze classifier head
        for p in self.baseline_model.get_classifier().parameters():
            p.requires_grad = True
        
    def set_bn_eval(self):
        for m in self.baseline_model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                m.track_running_stats = False

    def on_train_epoch_start(self):
        if self.hparams.freeze_backbone:
            print(self.hparams.freeze_backbone)
            self.set_bn_eval()



if __name__ == "__main__":
    model = TrashModel()
    x = torch.rand(1, 3, 224, 224)
    logger.info(f'Model: {model}')
    logger.info(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logger.info(f'Output shape of model: {model(x).shape}')

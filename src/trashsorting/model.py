from torch import nn
import torch
from pytorch_lightning import LightningModule
import timm
import logging

logger = logging.getLogger(__name__)

class TrashModel(LightningModule):
    def __init__(self, model_name="mobilenetv3_small_100", num_classes: int = 6, lr: float = 1e-3, pretrained: bool = True):
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

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.baseline_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.baseline_model.parameters())
        return torch.optim.Adam(params, lr=self.lr)

    def freeze_baseline_model(self):
        for p in self.baseline_model.parameters():
            p.requires_grad = False

    def unfreeze_baseline_model(self):
        for p in self.baseline_model.parameters():
            p.requires_grad = True


if __name__ == "__main__":
    model = TrashModel()
    x = torch.rand(1, 3, 224, 224)
    logger.info(f'Model: {model}')
    logger.info(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    logger.info(f'Output shape of model: {model(x).shape}')

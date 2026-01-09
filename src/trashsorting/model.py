from torch import nn
import torch
from pytorch_lightning import LightningModule
import timm


class TrashModel(LightningModule):
    def __init__(self, model_name="mobilenetv3_small_100", num_classes: int = 6, lr: float = 1e-3, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()
        # Create model from timm
        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=self.hparams.pretrained,
            num_classes=self.hparams.num_classes
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    model = TrashModel()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")

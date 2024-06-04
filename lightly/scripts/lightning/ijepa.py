import copy
import torch
import torchvision
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightly.data.collate import IJEPAMaskCollator
from lightly.models import utils
from lightly.models.modules.ijepa import IJEPABackbone, IJEPAPredictor
from lightly.transforms.ijepa_transform import IJEPATransform
from torchvision.datasets import VOCDetection

class IJEPA(pl.LightningModule):
    def __init__(self, vit_encoder, vit_predictor, momentum_scheduler, num_epochs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = IJEPABackbone.from_vit(vit_encoder)
        self.predictor = IJEPAPredictor.from_vit_encoder(
            vit_predictor.encoder, (vit_predictor.image_size // vit_predictor.patch_size) ** 2,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.momentum_scheduler = momentum_scheduler
        self.criterion = nn.SmoothL1Loss()

    def forward_target(self, imgs, masks_enc, masks_pred):
        with torch.no_grad():
            h = self.target_encoder(imgs)
            h = nn.functional.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            B = len(h)
            # -- create targets (masked regions of h)
            h = utils.apply_masks(h, masks_pred)
            h = utils.repeat_interleave_batch(h, B, repeat=len(masks_enc))
            return h

    def forward_context(self, imgs, masks_enc, masks_pred):
        z = self.encoder(imgs, masks_enc)
        z = self.predictor(z, masks_enc, masks_pred)
        return z

    def training_step(self, batch, batch_idx):
        udata, masks_enc, masks_pred = batch
        imgs, masks_enc, masks_pred = self.load_imgs(udata, masks_enc, masks_pred)
        z, h = self(imgs, masks_enc, masks_pred)
        loss = self.criterion(z, h)
        self.log("train_loss", loss)
        return loss

    def load_imgs(self, udata, masks_enc, masks_pred):
        imgs = udata[0]
        masks_1 = [u for u in masks_enc]
        masks_2 = [u for u in masks_pred]
        return imgs, masks_1, masks_2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        self.update_target_encoder()

    def update_target_encoder(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

def main():
    transform = IJEPATransform()
    collator = IJEPAMaskCollator(input_size=(224, 224), patch_size=32)

    #dataset = VOCDetection("datasets/pascal_voc", download=True, transform=transform, target_transform=lambda t: 0)
    dataset = torchvision.datasets.CIFAR10(
        "datasets/cifar10", download=True, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collator, batch_size=10, persistent_workers=False)

    vit_for_predictor = torchvision.models.vit_b_32(pretrained=False)
    vit_for_embedder = torchvision.models.vit_b_32(pretrained=False)
    ema = (0.996, 1.0)
    ipe_scale = 1.0
    ipe = len(data_loader)
    num_epochs = 10
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale)
        for i in range(int(ipe * num_epochs * ipe_scale) + 1)
    )

    model = IJEPA(vit_for_embedder, vit_for_predictor, momentum_scheduler, num_epochs, lr=1.5e-4)
    trainer = pl.Trainer(max_epochs=num_epochs, devices=1, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    trainer.fit(model, data_loader)

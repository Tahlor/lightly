# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import sys
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.data import LightlyDataset

class BarlowTwins(pl.LightningModule):
    def __init__(self):
        super().__init__()
        if False:
            resnet = torchvision.models.resnet18()
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        elif False:
            mobile_net = torchvision.models.mobilenet_v3_small()
            self.backbone = nn.Sequential(*list(mobile_net.children())[:-1])
            self.projection_head = BarlowTwinsProjectionHead(576, 2048, 2048)
        else:
            mobile_net = torchvision.models.mobilenet_v3_large()
            self.backbone = nn.Sequential(*list(mobile_net.children())[:-1])
            self.projection_head = BarlowTwinsProjectionHead(960, 2048, 2048)

        self.criterion = BarlowTwinsLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss', loss, on_step=True, prog_bar=True, on_epoch=True, batch_size=x0.shape[0])
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)
        return optim, scheduler

def main(dataset_path):
    model = BarlowTwins()

    # BarlowTwins uses BYOL augmentations.
    # We disable resizing and gaussian blur for cifar10.
    transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=448),
        view_2_transform=BYOLView2Transform(input_size=448),
    )
    # dataset = torchvision.datasets.CIFAR10(
    #     "datasets/cifar10", download=True, transform=transform
    # )
    dataset = LightlyDataset(dataset_path, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=200,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)

    # save the model
    torch.save(model.state_dict(), "barlowtwins.ckpt")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0])

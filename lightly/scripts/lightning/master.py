import sys
from lightly.scripts.lightning.barlowtwins import BarlowTwins
from lightly.scripts.lightning.swav import SwaV
from lightly.scripts.lightning.byol import BYOL
from lightly.scripts.lightning.ijepa import IJEPA
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
import pytorch_lightning as pl

def main(model_name, dataset_path):
    model_map = {
        'barlowtwins': BarlowTwins,
        'swav': SwaV,
        'byol': BYOL,
        'ijepa': IJEPA,
    }

    if model_name not in model_map:
        print(f"Error: {model_name} is not a valid model name.")
        sys.exit(1)

    ModelClass = model_map[model_name]
    model = ModelClass()

    # Common transformations and dataset preparation could be defined here
    # For demonstration, let's assume all use the same dataloader setup
    transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=448),
        view_2_transform=BYOLView2Transform(input_size=448),
    )

    # Replace this with the actual dataset for your environment
    dataset = torchvision.datasets.CIFAR10(
        dataset_path, download=True, transform=transform
    )

    # Assuming all models use a similar dataloader configuration
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=200,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=10, devices=1, accelerator=accelerator)
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Save the model
    torch.save(model.state_dict(), f"{model_name}.ckpt")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python master_script.py <model_name> <dataset_path>")
        sys.exit(1)

    model_name, dataset_path = sys.argv[1], sys.argv[2]
    main(model_name, dataset_path)

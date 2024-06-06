from pathlib import Path
import sys
import argparse
from lightly.scripts.lightning.barlowtwins import BarlowTwins
from lightly.scripts.lightning.swav import SwaV
from lightly.scripts.lightning.byol import BYOL
from lightly.scripts.lightning.ijepa import IJEPA
import torch
import torchvision
from torch import nn
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.data import LightlyDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from torch.utils.data import Subset

torch.set_float32_matmul_precision('medium')

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Self-supervised learning with various models.")
    parser.add_argument("--model_name", type=str, required=True, choices=['barlowtwins', 'swav', 'byol', 'ijepa'],
                        help="Name of the model to use.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for dataloader.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--input_size", type=int, default=448, help="Input size for the transforms.")
    parser.add_argument("--max_items", type=int, default=100, help="Maximum number of items to load from the dataset.")

    return parser.parse_args(args)


def main(args):
    model_map = {
        'barlowtwins': BarlowTwins,
        'swav': SwaV,
        'byol': BYOL,
        'ijepa': IJEPA,
    }
    model_save_path = f"{args.model_name}.ckpt"
    lightning_save_path = f"{args.model_name}_lightning.ckpt"

    ModelClass = model_map[args.model_name]

    if Path(lightning_save_path).is_file():
        model = ModelClass.load_from_checkpoint(lightning_save_path)
    else:
        model = ModelClass()

    transform = BYOLTransform(
        view_1_transform=BYOLView1Transform(input_size=args.input_size),
        view_2_transform=BYOLView2Transform(input_size=args.input_size),
    )

    dataset = LightlyDataset(args.dataset_path, transform=transform)
    indices = list(range(min(args.max_items, len(dataset))))
    dataset = Subset(dataset, indices)
    print(f"Dataset length: {len(dataset)}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    progress_bar = TQDMProgressBar(refresh_rate=5)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=1,
        accelerator=accelerator,
        callbacks=[lr_monitor, progress_bar],
        log_every_n_steps=50,
        default_root_dir=f"./{args.model_name}"
    )
    trainer.fit(model=model, train_dataloaders=dataloader)

    torch.save(model.state_dict(), model_save_path)
    trainer.save_checkpoint(lightning_save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)

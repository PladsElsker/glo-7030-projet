from dataclasses import dataclass
from pathlib import Path

import click
import mlflow
import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.models.transformer_backbone import MT5Backbone
from core.models.uni_sign import UniSign, UniSignModelParameters


@click.command()
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="Path to the dataset")
@click.option("--save-checkpoint-directory", "-s", required=True, type=str, help="Directory to save the checkpoints")
@click.option("--pretrained-encoder", "-pe", type=str, help="Path to the pretrained encoder weights")
@click.option("--transformer-type", "-t", type=click.Choice(["mt5", "t5"]), default="mt5", help="Type of transformer to use for language translation")
@click.option("--epochs", "-e", type=int, default=40, help="Amount of epochs to train")
@click.option("--batch-size", "-b", type=int, default=8, help="Batch size")
@click.option("--gradient-accumulation", "-g", type=int, default=1, help="Gradient accumulation")
@click.option("--num-workers", "-w", type=int, default=1, help="Amount of workers to use for dataset loading")
@click.option("--learning-rate", "-lr", type=float, default=3e-4, help="Learning rate")
@click.option("--device", "-dev", type=str, default="cuda", help="Torch device used for training")
def main(  # noqa: PLR0913
    dataset: str,
    save_checkpoint_directory: str,
    pretrained_encoder: str,
    transformer_type: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    num_workers: int,
    learning_rate: float,
    device: str,
) -> None:
    save_checkpoint_directory = Path(save_checkpoint_directory)

    translation_transformer = None

    if transformer_type == "mt5":
        translation_transformer = MT5Backbone(language="English")
        translation_transformer.load_model_and_tokenizer()
        for parameter in translation_transformer.parameters():
            parameter.requires_grad = False
    elif transformer_type == "t5":
        pass

    model_arguments = UniSignModelParameters(text_backbone=translation_transformer)
    model = UniSign(model_arguments)

    if pretrained_encoder is not None:
        pretrained_encoder = Path(pretrained_encoder)
        model.load_state_dict(torch.load(pretrained_encoder, weights_only=True), strict=False)

    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs // gradient_accumulation)

    train_dataset = TranslationDataset(dataset, train=True)
    test_dataset = TranslationDataset(dataset, test=True)

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    train_args = TrainingRunArguments(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=epochs,
        gradient_accumulation=gradient_accumulation,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device(device),
        save_dir=save_checkpoint_directory,
    )
    execute_training_run(train_args)


def execute_training_run(train_args: "TrainingRunArguments") -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5678")
    mlflow.set_experiment("SignTerpreter")

    train_args.model.to(train_args.device)

    for epoch_id in range(1, train_args.epochs + 1):
        logger.info(f"Epoch: {epoch_id}")

        train_loss = train_one_epoch(train_args)
        val_loss = validate_one_epoch(train_args)

        train_args.scheduler.step()

        mlflow.log_metric("train_loss", train_loss, step=epoch_id)
        mlflow.log_metric("val_loss", val_loss, step=epoch_id)

        checkpoint_path = train_args.save_dir / f"checkpoint_epoch_{epoch_id}.pt"
        torch.save(train_args.model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")


def train_one_epoch(train_args: "TrainingRunArguments") -> float:
    train_args.model.train()

    gradient_accumulation_counter = 0
    running_loss = 0.0
    steps = 0

    for x, y in tqdm(train_args.train_dataset, desc="Training"):
        x = x.to(train_args.device)
        y = y.to(train_args.device)

        out = train_args.model(x)
        loss = out["loss"]

        loss.backward()
        gradient_accumulation_counter += 1
        running_loss += loss.item()
        steps += 1

        if gradient_accumulation_counter >= train_args.gradient_accumulation:
            gradient_accumulation_counter = 0
            train_args.optimizer.step()
            train_args.optimizer.zero_grad()

    avg_loss = running_loss / steps
    logger.info(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


def validate_one_epoch(train_args: "TrainingRunArguments") -> float:
    train_args.model.eval()

    running_loss = 0.0
    steps = 0

    with torch.no_grad():
        for x, y in tqdm(train_args.test_dataset, desc="Validation"):
            x = x.to(train_args.device)
            y = y.to(train_args.device)

            out = train_args.model(x)
            loss = out["loss"]

            running_loss += loss.item()
            steps += 1

    avg_loss = running_loss / steps
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


@dataclass
class TrainingRunArguments:
    model: torch.nn.Module
    train_dataset: DataLoader
    test_dataset: DataLoader
    epochs: int
    gradient_accumulation: int
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    device: torch.device
    save_dir: Path


# class TranslationDataset(Dataset):
#     pass


# creation des fausses données pour tester à vider si tout va bien avc le script
# errors lors de l'execution /uni_Sign/config: module introuvable
class TranslationDataset(Dataset):
    def __init__(self, *, dataset_path: str | Path, train: bool = True, test: bool = False) -> None:  # noqa: ARG002
        self.length = 100

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple:
        x = torch.randn(1, 512)
        y = torch.randint(0, 2, (1,))
        return x, y


if __name__ == "__main__":
    main()

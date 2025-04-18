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
    if save_checkpoint_directory is not None:
        save_checkpoint_directory = Path(save_checkpoint_directory)

    translation_transformer = None
    if transformer_type == "mt5":
        translation_transformer = MT5Backbone(language="English")
        translation_transformer.load_model_and_tokenizer()
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

    execute_training_run(
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


def execute_training_run(  # noqa: PLR0913
    model: torch.nn.Module,
    train_dataset: DataLoader,
    test_dataset: DataLoader,  # noqa: ARG001
    epochs: int,
    gradient_accumulation: int,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    save_dir: Path,  # noqa: ARG001
) -> None:
    mlflow.set_tracking_uri("http://127.0.0.1:5678")
    mlflow.set_experiment("SignTerpreter")

    for epoch_id in range(1, epochs + 1):
        logger.info(f"Epoch: {epoch_id}")
        train_one_epoch(model, train_dataset, gradient_accumulation, optimizer, device)
        scheduler.step()


def train_one_epoch(
    model: torch.nn.Module,
    train_dataset: DataLoader,
    gradient_accumulation: int,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> None:
    gradient_accumulation_counter = 0
    for x, y in tqdm(train_dataset):
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        out["loss"].backward()
        gradient_accumulation_counter += 1
        if gradient_accumulation_counter >= gradient_accumulation:
            gradient_accumulation_counter = 0
            optimizer.step()


class TranslationDataset(Dataset):
    pass


if __name__ == "__main__":
    main()

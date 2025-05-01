import core.train.unisign_syspath  # noqa: F401, I001

import contextlib
import json
import pickle
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import click
import matplotlib.pyplot as plt
import mlflow
import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.models.transformer_backbone import MT5Backbone
from core.models.uni_sign import UniSign, UniSignModelParameters, collate_fn
from core.models.uni_sign import preprocess_data as preprocess_pkl_samples


@click.command()
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="Path to the dataset")
@click.option("--save-checkpoint-directory", "-s", required=True, type=str, help="Directory to save the checkpoints")
@click.option("--pretrained-encoder", "-pe", type=str, help="Path to the pretrained encoder weights")
@click.option("--transformer-type", "-t", type=click.Choice(["mt5", "t5"]), default="mt5", help="Type of transformer to use for language translation")
@click.option("--epochs", "-e", type=int, default=40, help="Amount of epochs to train")
@click.option("--batch-size", "-b", type=int, default=8, help="Batch size")
@click.option("--gradient-accumulation", "-g", type=int, default=1, help="Gradient accumulation")
@click.option("--num-workers", "-w", type=int, default=1, help="Amount of workers to use for dataset loading")
@click.option("--learning-rate", "-lr", type=float, default=1e-4, help="Learning rate")
@click.option("--device", "-dev", type=str, default="cuda", help="Torch device used for training")
@click.option("--mlflow", "-m", is_flag=True, default=False, help="Use mlflow instead of matplotlib for training analytics")
def train(  # noqa: PLR0913
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
    mlflow: bool,
) -> None:
    dataset = Path(dataset)
    save_checkpoint_directory = Path(save_checkpoint_directory)

    translation_transformer = None

    logger.info("Loading transformer model")
    if transformer_type == "mt5":
        translation_transformer = MT5Backbone(language="English")
        translation_transformer.load_model_and_tokenizer()
        for parameter in translation_transformer.parameters():
            parameter.requires_grad = False
    elif transformer_type == "t5":
        pass

    logger.info("Loading UniSign model")
    model_arguments = UniSignModelParameters(text_backbone=translation_transformer)
    model = UniSign(model_arguments)

    if pretrained_encoder is not None:
        pretrained_encoder = Path(pretrained_encoder)
        model.load_state_dict(torch.load(pretrained_encoder, weights_only=True), strict=False)

    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs // gradient_accumulation)

    train_dataset = TranslationDataset(dataset, split=DatasetSplit.TRAIN)
    test_dataset = TranslationDataset(dataset, split=DatasetSplit.TEST)

    train_dataset = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    test_dataset = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

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
        use_mlflow=mlflow,
    )
    execute_training_run(train_args)


def execute_training_run(train_args: "TrainingRunArguments") -> None:
    logger.info("Start experiment")
    if train_args.use_mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5678")
        mlflow.set_experiment("SignTerpreter")

    training_parameters = (
        {
            "epochs": train_args.epochs,
            "gradient_accumulation": train_args.gradient_accumulation,
            "optimizer": train_args.optimizer.__class__.__name__,
            "scheduler": train_args.scheduler.__class__.__name__,
            "device": str(train_args.device),
            "save_dir": str(train_args.save_dir),
        },
    )
    if train_args.use_mlflow:
        mlflow.start_run().__enter__()
        mlflow.log_params(training_parameters)
    else:
        with Path.open("trained_models/last_train_parameters.json", "w") as f:
            f.write(json.dumps(training_parameters))

    train_args.model.to(train_args.device)

    train_losses_per_epoch = []
    validation_losses_per_epoch = []

    for epoch_id in range(1, train_args.epochs + 1):
        logger.info(f"Epoch: {epoch_id}")

        train_loss = train_one_epoch(train_args, epoch_id)
        val_loss = validate_one_epoch(train_args, epoch_id)

        train_losses_per_epoch.append(train_loss)
        validation_losses_per_epoch.append(val_loss)

        train_args.scheduler.step()

        if train_args.use_mlflow:
            mlflow.log_metric("train_epoch_loss", train_loss, step=epoch_id)
            mlflow.log_metric("validation_epoch_loss", val_loss, step=epoch_id)
        else:
            plot_loss_per_epoch_plt(epoch_id, train_losses_per_epoch, validation_losses_per_epoch)

        checkpoint_path = train_args.save_dir / f"checkpoint_epoch_{epoch_id}.pt"
        torch.save(train_args.model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    if train_args.use_mlflow:
        mlflow.start_run().__exit__(None, None, None)


def train_one_epoch(train_args: "TrainingRunArguments", epoch_id: int) -> float:
    train_args.model.train()

    gradient_accumulation_counter = 0
    running_losses = []

    with tqdm(desc="Training", total=len(train_args.train_dataset)) as p_bar:
        for step_id, (x, y) in enumerate(train_args.train_dataset):
            for k, v in x.items():
                try:
                    x[k] = v.to(train_args.device).to(torch.float32)
                except AttributeError as e:
                    contextlib.suppress(e)

            gradient_accumulation_counter += 1

            if gradient_accumulation_counter >= train_args.gradient_accumulation:
                train_args.optimizer.zero_grad()

            loss = train_args.model(x, y)["loss"]
            loss.backward()
            running_losses.append(loss.item())

            if train_args.use_mlflow:
                global_step_id = step_id + len(train_args.train_dataset) * (epoch_id - 1)
                mlflow.log_metric("train_running_loss", loss.item(), step=global_step_id)

            if gradient_accumulation_counter >= train_args.gradient_accumulation:
                gradient_accumulation_counter = 0
                train_args.optimizer.step()

            p_bar.update(1)
            p_bar.set_description(f"Training | Loss: {sum(running_losses) / len(running_losses)}")

    avg_loss = sum(running_losses) / len(running_losses)
    logger.info(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


def validate_one_epoch(train_args: "TrainingRunArguments", epoch_id: int) -> float:
    train_args.model.eval()

    running_losses = []

    with torch.no_grad():
        for step_id, (x, y) in enumerate(tqdm(train_args.test_dataset, desc="Validation")):
            for k, v in x.items():
                try:
                    x[k] = v.to(train_args.device).to(torch.float32)
                except AttributeError as e:
                    contextlib.suppress(e)

            loss = train_args.model(x, y)["loss"]
            running_losses.append(loss.item())

            if train_args.use_mlflow:
                global_step_id = step_id + len(train_args.test_dataset) * (epoch_id - 1)
                mlflow.log_metric("validation_running_loss", loss.item(), step=global_step_id)

    avg_loss = sum(running_losses) / len(running_losses)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def plot_loss_per_epoch_plt(current_epoch: int, training_losses_per_epoch: list, validation_losses_per_epoch: list) -> None:
    _, axes = plt.subplots(1, 1, figsize=(12, 8))

    epochs_iterator = range(1, current_epoch + 1)
    axes.plot(epochs_iterator, training_losses_per_epoch, label="Train Loss", marker="o")
    axes.plot(epochs_iterator, validation_losses_per_epoch, label="Validation Loss", marker="o")

    axes.set_title("Loss per Epoch")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.legend()
    axes.grid()

    plt.tight_layout()
    plt.savefig("trained_models/loss_curves.png")


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
    use_mlflow: bool


class TranslationDataset(Dataset):
    def __init__(self, dataset_path: Path, split: "DatasetSplit") -> None:
        self.samples = list(dataset_path.glob("*.pkl"))
        self.idx_offset = 0

        start_test_index = int(len(self.samples) * 0.8)
        start_validation_index = int(len(self.samples) * 0.9)

        if split == DatasetSplit.TRAIN:
            self.samples = self.samples[:start_test_index]
        elif split == DatasetSplit.TEST:
            self.samples = self.samples[start_test_index:start_validation_index]
        elif split == DatasetSplit.VALIDATION:
            self.samples = self.samples[start_validation_index:]

        logger.info(f"Loaded {len(self.samples)} samples for {split.name} dataset")

    def __len__(self) -> int:
        return len(self.samples) - self.idx_offset

    def __getitem__(self, idx: int) -> tuple:
        with Path.open(self.samples[idx + self.idx_offset], "rb") as pkl_file:
            sample = pickle.load(pkl_file)

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(sample["poses"], f)
            temp_path = Path(f.name)
        try:
            res = (
                *preprocess_pkl_samples(
                    temp_path,
                    sample["raw_text"],
                ),
            )
            temp_path.unlink(missing_ok=True)
        except ValueError:
            self.idx_offset += 1
            res = self.__getitem__(idx)

        return res


def collate_function(samples: list) -> tuple:
    x_batch = [sample[0] for sample in samples]
    y_batch = [sample[1] for sample in samples]
    return x_batch[0], y_batch[0]


class DatasetSplit(Enum):
    TEST = 0
    TRAIN = 1
    VALIDATION = 2


if __name__ == "__main__":
    train()

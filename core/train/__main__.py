import contextlib
import pickle
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sacrebleu

import click
import mlflow
import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.models.transformer_backbone import MT5Backbone
from core.models.uni_sign import UniSign, UniSignModelParameters, collate_fn, preprocess_data as preprocess_pkl_samples

sys.path[0:0] = ["uni_sign"]


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

    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999))
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
    )
    execute_training_run(train_args)


def execute_training_run(train_args: "TrainingRunArguments") -> None:
    logger.info("Start experiment")
    
    use_mlflow = False
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5678")
        mlflow.set_experiment("SignTerpreter")
        use_mlflow = True
        logger.info("MLflow tracking enabled")
    except Exception as e:
        logger.warning(f"MLflow tracking disabled: {e}")
        logger.warning("Continuing training without MLflow tracking")
    
    if use_mlflow:
        with mlflow.start_run():
            for python_file in Path().glob("*.py"):
                mlflow.log_artifact(python_file, artifact_path="source_code")

            mlflow.log_params(
                {
                    "epochs": train_args.epochs,
                    "gradient_accumulation": train_args.gradient_accumulation,
                    "optimizer": train_args.optimizer.__class__.__name__,
                    "scheduler": train_args.scheduler.__class__.__name__,
                    "device": str(train_args.device),
                    "save_dir": str(train_args.save_dir),
                },
            )
            
            _run_training(train_args, use_mlflow)
    else:
        _run_training(train_args, use_mlflow)

def _run_training(train_args: "TrainingRunArguments", use_mlflow: bool) -> None:
    train_args.model.to(train_args.device)

    for epoch_id in range(1, train_args.epochs + 1):
        logger.info(f"Epoch: {epoch_id}")

        train_loss = train_one_epoch(train_args, epoch_id, use_mlflow)
        val_loss, bleu_scores = validate_one_epoch(train_args, epoch_id, use_mlflow)

        train_args.scheduler.step()

        if use_mlflow:
            mlflow.log_metric("train_epoch_loss", train_loss, step=epoch_id)
            mlflow.log_metric("validation_epoch_loss", val_loss, step=epoch_id)
            
            for metric_name, score in bleu_scores.items():
                mlflow.log_metric(metric_name, score, step=epoch_id)

        checkpoint_path = train_args.save_dir / f"checkpoint_epoch_{epoch_id}.pt"
        torch.save(train_args.model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")


def train_one_epoch(train_args: "TrainingRunArguments", epoch_id: int, use_mlflow: bool = True) -> float:
    train_args.model.train()

    gradient_accumulation_counter = 0
    running_losses = []

    train_args.optimizer.zero_grad()

    with tqdm(desc="Training", total=len(train_args.train_dataset)) as p_bar:
        for step_id, (x, y) in enumerate(train_args.train_dataset):
            if gradient_accumulation_counter == 0:
                train_args.optimizer.zero_grad()
            
            for k, v in x.items():
                try:
                    x[k] = v.to(train_args.device).to(torch.float32)
                except AttributeError as e:
                    contextlib.suppress(e)
            loss = train_args.model(x, y)["loss"] 
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(train_args.model.parameters(), max_norm=1.0)
            
            gradient_accumulation_counter += 1
            running_losses.append(loss.item())

            if use_mlflow:
                global_step_id = step_id + len(train_args.train_dataset) * (epoch_id - 1)
                mlflow.log_metric("train_running_loss", loss.item(), step=global_step_id)

            if gradient_accumulation_counter >= train_args.gradient_accumulation:
                train_args.optimizer.step()
                train_args.optimizer.zero_grad()
                gradient_accumulation_counter = 0

            p_bar.update(1)
            p_bar.set_description(f"Training | Loss: {sum(running_losses[-100:]) / min(len(running_losses), 100):.8f}")

    if gradient_accumulation_counter > 0:
        train_args.optimizer.step()
        train_args.optimizer.zero_grad()

    avg_loss = sum(running_losses) / len(running_losses) if running_losses else float('inf')
    logger.info(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


def validate_one_epoch(train_args: "TrainingRunArguments", epoch_id: int, use_mlflow: bool = True) -> tuple[float, dict]:
    train_args.model.eval()

    running_losses = []
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for step_id, (x, y) in enumerate(tqdm(train_args.test_dataset, desc="Validation")):
            for k, v in x.items():
                try:
                    x[k] = v.to(train_args.device).to(torch.float32)
                except AttributeError:
                    continue

            outputs = train_args.model(x, y)
            loss = outputs["loss"]
            
            predictions = outputs.get("predictions", [])
            
            if not predictions and "gt_sentence" in y:
                all_references.extend(y["gt_sentence"])
                all_predictions.extend([""] * len(y["gt_sentence"]))
            
            running_losses.append(loss.item())
            
            if predictions:
                all_predictions.extend(predictions)
            
            if use_mlflow:
                global_step_id = step_id + len(train_args.test_dataset) * (epoch_id - 1)
                mlflow.log_metric("validation_running_loss", loss.item(), step=global_step_id)

    avg_loss = sum(running_losses) / len(running_losses) if running_losses else 0.0
    
    bleu_scores = {}
    if all_predictions and all_references:
        try:
            bleu_scores['bleu'] = sacrebleu.corpus_bleu(all_predictions, [all_references]).score
            bleu_scores['bleu-2'] = sacrebleu.corpus_bleu(all_predictions, [all_references], max_ngram_order=2).score
            bleu_scores['bleu-3'] = sacrebleu.corpus_bleu(all_predictions, [all_references], max_ngram_order=3).score
            bleu_scores['bleu-4'] = sacrebleu.corpus_bleu(all_predictions, [all_references], max_ngram_order=4).score
            
            logger.info(f"Validation Loss: {avg_loss:.4f}")
            logger.info(f"BLEU Score: {bleu_scores['bleu']:.2f}")
            logger.info(f"BLEU-2 Score: {bleu_scores['bleu-2']:.2f}")
            logger.info(f"BLEU-3 Score: {bleu_scores['bleu-3']:.2f}")
            logger.info(f"BLEU-4 Score: {bleu_scores['bleu-4']:.2f}")
            
            if use_mlflow:
                for metric_name, score in bleu_scores.items():
                    mlflow.log_metric(metric_name, score, step=epoch_id)
        except Exception:
            logger.warning("Erreur lors du calcul des scores BLEU")
    else:
        logger.warning("Pas de prédictions ou références disponibles pour calculer le score BLEU")

    return avg_loss, bleu_scores


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


class TranslationDataset(Dataset):
    def __init__(self, dataset_path: Path, split: "DatasetSplit") -> None:
        all_samples = list(dataset_path.glob("*.pkl"))
        
        if split == DatasetSplit.TRAIN:
            self.samples = all_samples[:int(len(all_samples) * 0.8)]
        elif split == DatasetSplit.TEST:
            self.samples = all_samples[int(len(all_samples) * 0.8):int(len(all_samples) * 0.9)]
        elif split == DatasetSplit.VALIDATION:
            self.samples = all_samples[int(len(all_samples) * 0.9):]
        
        self.split = split
        logger.info(f"Loaded {len(self.samples)} samples for {split.name} dataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        with Path.open(self.samples[idx], "rb") as pkl_file:
            sample = pickle.load(pkl_file)

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(sample["poses"], f)
            temp_path = Path(f.name)

        src_input, tgt_input = preprocess_pkl_samples(
            temp_path,
            sample["raw_text"],
        )
        
        temp_path.unlink(missing_ok=True)
        
        return src_input, tgt_input


class DatasetSplit(Enum):
    TEST = 0
    TRAIN = 1
    VALIDATION = 2


if __name__ == "__main__":
    train()

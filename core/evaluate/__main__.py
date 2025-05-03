import core.train.unisign_syspath  # noqa: F401, I001

import pickle
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from tqdm import tqdm
import contextlib
import numpy as np

import click
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from uni_sign.SLRT_metrics import translation_performance

from core.models.transformer_backbone import MT5Backbone
from core.models.uni_sign import UniSign, UniSignModelParameters, collate_fn
from core.models.uni_sign import preprocess_data as preprocess_pkl_samples


@click.command()
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True), help="Path to the dataset")
@click.option("--pretrained", "-pe", required=True, type=str, help="Path to the pretrained encoder weights")
@click.option("--transformer-type", "-t", type=click.Choice(["mt5", "t5"]), default="mt5", help="Type of transformer to use for language translation")
@click.option("--batch-size", "-b", type=int, default=8, help="Batch size")
@click.option("--num-workers", "-w", type=int, default=1, help="Amount of workers to use for dataset loading")
@click.option("--device", "-dev", type=str, default="cuda", help="Torch device used for training")
def evaluate(
    dataset: str,
    pretrained: str,
    transformer_type: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> None:
    dataset = Path(dataset)

    translation_transformer = None

    logger.info("Loading transformer model")
    if transformer_type == "mt5":
        translation_transformer = MT5Backbone(language="English")
        translation_transformer.load_model_and_tokenizer()
    elif transformer_type == "t5":
        pass

    logger.info("Loading UniSign model")
    model_arguments = UniSignModelParameters(text_backbone=translation_transformer)
    model = UniSign(model_arguments)

    if pretrained is not None:
        pretrained = Path(pretrained)
        model.load_state_dict(torch.load(pretrained, weights_only=True), strict=False)

    for parameter in model.parameters():
        parameter.requires_grad = False

    dataset = TranslationDataset(dataset, split=DatasetSplit.TRAIN)

    dataset = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    evaluation_args = EvaluationRunArguments(
        model=model,
        dataset=dataset,
        device=torch.device(device),
    )
    execute_evaluation(evaluation_args)


def execute_evaluation(evaluation_args: "EvaluationRunArguments") -> None:
    logger.info("Start evaluation")

    evaluation_args.model.to(evaluation_args.device)
    evaluation_args.model.eval()

    results = []

    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(evaluation_args.dataset, desc="Test")):
            for k, v in x.items():
                try:
                    x[k] = v.to(evaluation_args.device).to(torch.float32)
                except AttributeError as e:
                    contextlib.suppress(e)

            output = evaluation_args.model(x, y)
            predictions = evaluation_args.model.generate(output, max_new_tokens=50, num_beams=2)

            tgt_pres = []
            tgt_refs = []

            for i in range(len(predictions)):
                if hasattr(evaluation_args.model, "transformer") and hasattr(evaluation_args.model.transformer, "tokenizer"):
                    pred_text = evaluation_args.model.transformer.tokenizer.decode(predictions[i], skip_special_tokens=True)

                    ref_text = y["gt_sentence"] if (isinstance(y, dict) and "gt_sentence" in y) else str(y)
                    tgt_pres.append(pred_text)
                    tgt_refs.append(ref_text)

            if tgt_pres and tgt_refs:
                if isinstance(tgt_refs[0], list):
                    tgt_refs = [ref[0] if isinstance(ref, list) and len(ref) > 0 else str(ref) for ref in tgt_refs]

                bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
                logger.info(
                    f"""BLEU:
                    1: {bleu_dict.get('bleu1', 0):.2f}
                    2: {bleu_dict.get('bleu2', 0):.2f}
                    3: {bleu_dict.get('bleu3', 0):.2f}
                    4: {bleu_dict.get('bleu4', 0):.2f}
                    Rouge: {rouge_score:.2f}
                    """,
                )
                logger.info(f"Sample: '{tgt_pres[0]}'")

                results.append({"bleu": bleu_dict, "rouge": rouge_score})

        bleu1 = np.mean([result["bleu"]["bleu1"] for result in results])
        bleu2 = np.mean([result["bleu"]["bleu2"] for result in results])
        bleu3 = np.mean([result["bleu"]["bleu3"] for result in results])
        bleu4 = np.mean([result["bleu"]["bleu4"] for result in results])
        rouge = np.mean([result["rouge"] for result in results])

        logger.info("Mean scores:")
        logger.info(f"BLEU: 1: {bleu1:.6f} | 2: {bleu2:.6f} | 3: {bleu3:.6f} | 4: {bleu4:.6f} | Rouge: {rouge:.6f}")


@dataclass
class EvaluationRunArguments:
    model: torch.nn.Module
    dataset: DataLoader
    device: torch.device


class TranslationDataset(Dataset):
    def __init__(self, dataset_path: Path, split: "DatasetSplit") -> None:
        self.samples = list(dataset_path.glob("*.pkl"))[:16]
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
        if idx >= len(self):
            raise StopIteration

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


class DatasetSplit(Enum):
    TEST = 0
    TRAIN = 1
    VALIDATION = 2


if __name__ == "__main__":
    evaluate()

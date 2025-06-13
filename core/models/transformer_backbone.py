from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import MT5ForConditionalGeneration, T5Tokenizer


class BaseTransformerBackbone(nn.Module, ABC):
    def __init__(self, label_smoothing: float, max_length: int) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.max_length = max_length
        self.pad_token = -100
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=self.pad_token)

    @abstractmethod
    def load_model_and_tokenizer(self, path: Optional[Path] = None) -> Tuple[nn.Module, nn.Module]:
        pass

    @abstractmethod
    def tokenize_labels(self, sentences: list) -> torch.Tensor:
        pass

    @abstractmethod
    def translate_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> any:
        pass

    def augment_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return embeddings, attention_mask

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sentences: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        augmented_embeddings, augmented_attention_mask = self.augment_embeddings(embeddings, attention_mask)
        labels = self.tokenize_labels(sentences).to(embeddings.device) if sentences else None
        outputs = self.translate_embeddings(inputs_embeds=augmented_embeddings, attention_mask=augmented_attention_mask, labels=labels)
        loss = self.compute_loss(outputs.logits, labels) if labels is not None else None
        return {
            "loss": loss,
            "inputs_embeds": augmented_embeddings,
            "attention_mask": augmented_attention_mask,
        }

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.permute(0, 2, 1)
        return self.criterion(logits, labels)


class MT5Backbone(BaseTransformerBackbone):
    def __init__(self, label_smoothing: float = 0.2, max_length: int = 50, language: str = "Chinese") -> None:
        super().__init__(label_smoothing=label_smoothing, max_length=max_length)
        self.language = language

    def load_model_and_tokenizer(
        self,
        path: Path = Path("uni_sign/pretrained_weight/mt5-base"),
    ) -> Tuple[nn.Module, nn.Module]:
        self.model = MT5ForConditionalGeneration.from_pretrained(str(path))
        self.tokenizer = T5Tokenizer.from_pretrained(str(path), legacy=True)

    def tokenize_labels(self, sentences: list) -> torch.Tensor:
        tokenized = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = tokenized["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def translate_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> any:
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def augment_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = embeddings.size(0)
        prompt = [f"Translate sign language video to {self.language}: "] * batch_size
        prompt_tokens = self.tokenizer(prompt, padding="longest", truncation=True, return_tensors="pt", max_length=self.max_length).to(
            embeddings.device,
        )
        prompt_embeddings = self.model.encoder.embed_tokens(prompt_tokens["input_ids"])
        augmented_embeddings = torch.cat([prompt_embeddings, embeddings], dim=1)
        augmented_attention_mask = torch.cat([prompt_tokens["attention_mask"], attention_mask], dim=1)
        return augmented_embeddings, augmented_attention_mask

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from transformers import MT5ForConditionalGeneration, MT5Tokenizer


class TransformerModelConfig:
    def __init__(self, label_smoothing: float, max_length: int = 50) -> None:
        self.label_smoothing = label_smoothing
        self.max_length = max_length


class BaseTransformerBackbone(nn.Module, ABC):
    def __init__(self, config: TransformerModelConfig, language: str) -> None:
        super().__init__()
        self.config = config
        self.language = language
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    @abstractmethod
    def load_model_and_tokenizer(self) -> Tuple[nn.Module, nn.Module]:
        pass

    def tokenize_labels(self, sentences: list) -> torch.Tensor:
        tokenized = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        labels = tokenized["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    def augmented_embedding(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return embeddings, attention_mask

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        sentences: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        augmented_embeddings, augmented_attention_mask = self.augmented_embedding(embeddings, attention_mask)
        labels = self.tokenize_labels(sentences).to(embeddings.device) if sentences else None
        outputs = self.model(
            inputs_embeds=augmented_embeddings,
            attention_mask=augmented_attention_mask,
            labels=labels,
            return_dict=True,
        )
        loss = self.compute_loss(outputs.logits, labels) if labels is not None else None
        return {
            "loss": loss,
            "inputs_embeds": augmented_embeddings,
            "attention_mask": augmented_attention_mask,
        }

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss_function = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing, ignore_index=-100)
        flattened_logits = logits.reshape(-1, logits.size(-1))
        flattened_labels = labels.reshape(-1)
        return loss_function(flattened_logits, flattened_labels)


class MT5Backbone(BaseTransformerBackbone):
    def load_model_and_tokenizer(self) -> Tuple[nn.Module, nn.Module]:
        path = "uni_sign/pretrained_weight/mt5-base"
        model = MT5ForConditionalGeneration.from_pretrained(path)
        tokenizer = MT5Tokenizer.from_pretrained(path)
        return model, tokenizer

    def augmented_embedding(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = embeddings.size(0)
        prompt = [f"Translate sign language video to {self.language}: "] * batch_size
        prompt_tokens = self.tokenizer(prompt, padding="longest", truncation=True, return_tensors="pt").to(embeddings.device)
        prompt_embeddings = self.model.encoder.embed_tokens(prompt_tokens["input_ids"])
        augmented_embeddings = torch.cat([prompt_embeddings, embeddings], dim=1)
        augmented_attention_mask = torch.cat([prompt_tokens["attention_mask"], attention_mask], dim=1)
        return augmented_embeddings, augmented_attention_mask

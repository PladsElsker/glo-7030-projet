import os
import random
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.io as io
from transformers import M2M100Tokenizer


VIDEO_ID = 'VIDEO_ID'
VIDEO_NAME = 'VIDEO_NAME'
SENTENCE_ID = 'SENTENCE_ID'
SENTENCE_NAME = 'SENTENCE_NAME'
START = 'START'
END = 'END'
START_REALIGNED = 'START_REALIGNED'
END_REALIGNED = 'END_REALIGNED'
SENTENCE = 'SENTENCE'


tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")


class How2SignDataset(Dataset):
    def __init__(self, video_path, csv_config, data_augmentation=None, target_processor=None):
        self.video_paths = {Path(file).stem: Path(video_path).resolve() / file for file in os.listdir(video_path)}
        self.csv_config = pd.read_csv(csv_config, delimiter='\t')
        self.data_augmentation = data_augmentation
        self.target_processor = target_processor
        self.iterator_offset = 0

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        idx += self.iterator_offset
        video_path = None

        while video_path is None:
            try:
                sentence_name = self.csv_config[SENTENCE_NAME][idx]
                video_path = self.video_paths[sentence_name]
            except KeyError:
                idx += 1
                self.iterator_offset += 1

        video, audio, info = io.read_video(video_path, pts_unit="sec")
        video = video.permute(0, 3, 1, 2)

        if self.data_augmentation:
            video = self.data_augmentation(video)

        target = self.csv_config[SENTENCE][idx]

        if self.target_processor:
            target = self.target_processor(target)

        return video, target


class RandomHorizontalFlipVideo:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, video):
        return torch.flip(video, dims=[3]) if random.random() < self.p else video


def m2m100_target_processor(text, max_length=512, tgt_lang='en_XX'):
    tokenizer.tgt_lang = tgt_lang
    return tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )


def collate_how2sign(batch, pad_token_id=tokenizer.pad_token_id):
    videos, targets = zip(*batch)
    max_frames = max(video.shape[0] for video in videos)
    padded_videos = []

    for video in videos:
        diff = max_frames - video.shape[0]
        pad = torch.zeros(diff, *video.shape[1:], dtype=video.dtype) if diff > 0 else None
        padded_videos.append(torch.cat([video, pad], dim=0) if pad is not None else video)

    padded_videos = torch.stack(padded_videos)
    input_ids_list = [t["input_ids"].squeeze(0) for t in targets]
    attention_mask_list = [t["attention_mask"].squeeze(0) for t in targets]
    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded_input_ids, padded_attention_masks = [], []

    for ids, mask in zip(input_ids_list, attention_mask_list):
        pad_length = max_len - ids.shape[0]
        padded_input_ids.append(torch.cat([ids, torch.full((pad_length,), pad_token_id, dtype=torch.long)]))
        padded_attention_masks.append(torch.cat([mask, torch.zeros(pad_length, dtype=torch.long)]))

    padded_input_ids = torch.stack(padded_input_ids)
    padded_attention_masks = torch.stack(padded_attention_masks)
    padded_targets = {"input_ids": padded_input_ids, "attention_mask": padded_attention_masks}
    return padded_videos, padded_targets

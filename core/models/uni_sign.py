from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.functional import pad
from torchvision import transforms

from uni_sign.datasets import S2T_Dataset_news
from uni_sign.models import Uni_Sign as BaseUniSign


@dataclass
class UniSignModelParameters:
    hidden_dim: int = 256
    dataset: str = "CSL_News"
    label_smoothing: float = 0.2
    rgb_support: bool = False
    text_backbone: nn.Module = None


def preprocess_data(pkl_path: Path, text: str, max_frames: int = 256) -> tuple:
    preproc = S2T_Dataset_news.load_pose
    postproc = S2T_Dataset_news.collate_fn
    s = SimpleNamespace()
    s.pose_dir = pkl_path.parent
    s.rgb_dir = ""
    s.max_length = max_frames
    s.data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ],
    )
    pkl_filename = pkl_path.name
    rgb_filename = ""
    s.rgb_support = False

    pose, rgb = preproc(s, pkl_filename, rgb_filename)
    return (*postproc(s, [[pkl_path.stem, pose, text, -1, rgb]]),)


def collate_fn(batches: list) -> tuple:
    src_input = {}
    tgt_input = {}

    keys = batches[0][0].keys()
    for key in keys:
        if key == "attention_mask":
            max_length = max(len(batch[0]["attention_mask"][0]) for batch in batches)
            src_input[key] = torch.stack(
                [pad(batch[0][key].squeeze(0), (0, max_length - batch[0][key].shape[1]), "constant", 0) for batch in batches],
            )
        elif isinstance(batches[-1][0][key], torch.Tensor):
            max_length = max((len(batch[0][key][0]) if len(batch[0][key].shape) > 1 else len(batch[0][key])) for batch in batches)
            src_input[key] = torch.stack(
                [pad_with_last(batch[0][key].squeeze(0) if len(batch[0][key].shape) > 1 else batch[0][key], max_length) for batch in batches],
            )
        else:
            src_input[key] = [batch[0][key][0] for batch in batches]

    keys = batches[0][1].keys()
    for key in keys:
        tgt_input[key] = [batch[1][key][0] for batch in batches]

    return src_input, tgt_input


def pad_with_last(data: torch.Tensor, length: int) -> torch.Tensor:
    if data.size(0) >= length:
        return data

    pad = data[-1].expand(length - data.size(0), *data.shape[1:])
    return torch.cat((data, pad), dim=0)


class UniSign(BaseUniSign):
    def __init__(self, args: UniSignModelParameters) -> None:
        super().__init__(args)
        self.args = args
        self.transformer = args.text_backbone

    def forward(self, src_input: dict, tgt_input: dict) -> dict:
        features = []
        body_feat = None
        for part in self.modes:
            proj_feat = self.proj_linear[part](src_input[part]).permute(0, 3, 1, 2)
            gcn_feat = self.gcn_modules[part](proj_feat)
            if part == "body":
                body_feat = gcn_feat
            else:
                if body_feat is None:
                    raise ValueError
                if part == "left":
                    gcn_feat = gcn_feat + body_feat[..., -2][..., None].detach()
                elif part == "right":
                    gcn_feat = gcn_feat + body_feat[..., -1][..., None].detach()
                elif part == "face_all":
                    gcn_feat = gcn_feat + body_feat[..., 0][..., None].detach()
                else:
                    raise NotImplementedError
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)
            pool_feat = gcn_feat.mean(-1).transpose(1, 2)
            features.append(pool_feat)

        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        inputs_embeds = self.pose_proj(inputs_embeds)

        return self.transformer(
            embeddings=inputs_embeds,
            attention_mask=src_input["attention_mask"],
            sentences=tgt_input["gt_sentence"],
        )

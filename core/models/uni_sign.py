from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torchvision import transforms

from core.models.transformer_backbone import MT5Backbone, TransformerModelConfig
from uni_sign.datasets import S2T_Dataset_news
from uni_sign.models import Uni_Sign as BaseUniSign


@dataclass
class UniSignModelParameters:
    hidden_dim: int = 256
    dataset: str = "CSL_News"
    label_smoothing: float = 0.2
    rgb_support: bool = False
    language: str = "Chinese"
    text_backbone: nn.Module = None

    def __post_init__(self) -> None:
        if self.text_backbone is None:
            config = TransformerModelConfig(label_smoothing=self.label_smoothing)
            self.text_backbone = MT5Backbone(config, self.language)


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
    return postproc(s, [[pkl_path.stem, pose, text, -1, rgb]])


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

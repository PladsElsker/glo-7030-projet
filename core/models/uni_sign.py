from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision import transforms

from uni_sign.datasets import S2T_Dataset_news
from uni_sign.models import Uni_Sign as BaseUniSign


pretrained_weights = {
    "Stage1": "uni_sign/pretrained_weight/Uni-Sign/csl_stage1_weight.pth",
    "Stage2": "uni_sign/pretrained_weight/Uni-Sign/csl_stage2_weight.pth",
}


@dataclass
class UniSignModelParameters:
    hidden_dim: int = 256
    dataset: str = "CSL_News"
    label_smoothing: float = 0.2
    rgb_support: bool = False


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
    def forward(self, src_input: dict, tgt_input: dict) -> dict:
        # Pose branch forward
        features = []

        body_feat = None
        for part in self.modes:
            # project position to hidden dim
            proj_feat = self.proj_linear[part](src_input[part]).permute(0, 3, 1, 2)  # B,C,T,V
            # spatial gcn forward
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

            # temporal gcn forward
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)  # B,C,T,V
            pool_feat = gcn_feat.mean(-1).transpose(1, 2)  # B,T,C
            features.append(pool_feat)

        # concat sub-pose feature across token dimension
        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        inputs_embeds = self.pose_proj(inputs_embeds)

        prefix_token = self.mt5_tokenizer(
            [f"Translate sign language video to {self.lang}: "] * len(tgt_input["gt_sentence"]),
            padding="longest",
            truncation=True,
            return_tensors="pt",
        ).to(inputs_embeds.device)

        prefix_embeds = self.mt5_model.encoder.embed_tokens(prefix_token["input_ids"])
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        attention_mask = torch.cat([prefix_token["attention_mask"], src_input["attention_mask"]], dim=1)

        tgt_input_tokenizer = self.mt5_tokenizer(tgt_input["gt_sentence"], return_tensors="pt", padding=True, truncation=True, max_length=50)

        labels = tgt_input_tokenizer["input_ids"]
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100

        out = self.mt5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels.to(inputs_embeds.device),
            return_dict=True,
        )

        label = labels.reshape(-1)
        out_logits = out["logits"]
        logits = out_logits.reshape(-1, out_logits.shape[-1])
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing, ignore_index=-100)
        loss = loss_fct(logits, label.to(out_logits.device, non_blocking=True))

        return {
            # use for inference
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "loss": loss,
        }

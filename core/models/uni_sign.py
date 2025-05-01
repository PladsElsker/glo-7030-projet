from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import pickle
import logging

import torch
from torch import nn
from torchvision import transforms

from uni_sign.datasets import S2T_Dataset_news
from uni_sign.models import Uni_Sign as BaseUniSign
from core.models.tensor_utils import (
    determine_max_frames,
    handle_attention_mask,
    process_pose_parts,
    process_target_data,
    load_part_kp,
    EXPECTED_KEYPOINTS
)

logger = logging.getLogger(__name__)

@dataclass
class UniSignModelParameters:
    hidden_dim: int = 256
    dataset: str = "Open_ASL"
    label_smoothing: float = 0.2
    rgb_support: bool = False
    text_backbone: nn.Module = None


def preprocess_data(pkl_path: Path, text: str, max_frames: int = 256) -> tuple:
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        if 'poses' in data and isinstance(data['poses'], dict):
            poses_dict = data['poses']
            
            if 'keypoints' in poses_dict and 'scores' in poses_dict:
                keypoints = poses_dict['keypoints']
                scores = poses_dict['scores']
                
                kps_with_scores = load_part_kp(keypoints, scores, force_ok=True)
                
                src_input = {}
                for part, tensor in kps_with_scores.items():
                    src_input[part] = tensor.unsqueeze(0)
                
                seq_len = tensor.shape[0]
                src_input['attention_mask'] = torch.ones((1, seq_len), dtype=torch.long)
                src_input['src_lengths'] = torch.tensor([seq_len], dtype=torch.long)
                
                tgt_input = {'gt_sentence': [text if text else ""]}
                
                return src_input, tgt_input
    except Exception as e:
        return _create_empty_data(text)
        
    return _create_empty_data(text)


def _create_empty_data(text):
    empty_src = {}
    empty_tgt = {}
    empty_src['attention_mask'] = torch.ones((1, 1), dtype=torch.float32)
    empty_src['src_lengths'] = torch.tensor([1], dtype=torch.long)
    
    for part, keypoints in EXPECTED_KEYPOINTS.items():
        empty_src[part] = torch.zeros((1, 1, keypoints, 3), dtype=torch.float32)
        
    empty_tgt['gt_sentence'] = [text if text else ""]
    
    return empty_src, empty_tgt


def collate_fn(batch: list) -> tuple:
    if not batch:
        return _create_empty_data("")
        
    filtered_batch = []
    for item in batch:
        if not item or len(item) < 2:
            continue
        
        src_dict, tgt_dict = item
        
        if isinstance(src_dict, dict):
            if 'keypoints' in src_dict and 'scores' in src_dict:
                filtered_batch.append(item)
            elif any(part in src_dict for part in ['body', 'left', 'right', 'face_all']):
                filtered_batch.append(item)
    
    if not filtered_batch:
        return _create_empty_data("")
        
    required_parts = ['body', 'left', 'right', 'face_all']
    max_frames = determine_max_frames(filtered_batch, required_parts)
    
    src_input, filtered_batch = process_pose_parts(filtered_batch, required_parts, max_frames)
    
    batch_size = len(filtered_batch)
    
    if batch_size == 0:
        return _create_empty_data("")
    
    src_input = handle_attention_mask(filtered_batch, src_input, max_frames, batch_size)
    src_input['src_lengths'] = torch.tensor([max_frames] * batch_size, dtype=torch.long)

    tgt_input = process_target_data(filtered_batch)
    
    return src_input, tgt_input


class UniSign(BaseUniSign):
    def __init__(self, args: UniSignModelParameters) -> None:
        super().__init__(args)
        self.args = args
        self.transformer = args.text_backbone

def forward(self, src_input: dict, tgt_input: dict) -> dict:
    if not src_input or not tgt_input or not tgt_input.get('gt_sentence'):
        return {"loss": torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)}
        
    features = []
    body_feat = None

    try:
        for part in self.modes:
            if part not in src_input:
                continue
                
            tensor = src_input[part]
            
            if tensor.dim() == 5:
                tensor = tensor.squeeze(1).squeeze(1)
            elif tensor.dim() == 4:
                tensor = tensor.squeeze(1)
            
            src_input[part] = tensor
            
            proj_feat = self.proj_linear[part](tensor).permute(0, 3, 1, 2)
            gcn_feat = self.gcn_modules[part](proj_feat)
            
            if part == "body":
                body_feat = gcn_feat
            else:
                if body_feat is None:
                    continue
                    
                if part == "left":
                    gcn_feat = gcn_feat + body_feat[..., -2][..., None].detach()
                elif part == "right":
                    gcn_feat = gcn_feat + body_feat[..., -1][..., None].detach()
                elif part == "face_all":
                    gcn_feat = gcn_feat + body_feat[..., 0][..., None].detach()
                
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat)
            pool_feat = gcn_feat.mean(-1).transpose(1, 2)
            features.append(pool_feat)
        
        if not features:
            return {"loss": torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)}
            
        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        inputs_embeds = self.pose_proj(inputs_embeds)
        
        if 'attention_mask' in src_input and src_input['attention_mask'].dim() == 2:
            attention_mask = src_input['attention_mask']
        else:
            batch_size = inputs_embeds.size(0)
            seq_len = inputs_embeds.size(1)
            attention_mask = torch.ones((batch_size, seq_len), device=inputs_embeds.device)
        
        return self.transformer(
            embeddings=inputs_embeds,
            attention_mask=attention_mask,
            sentences=tgt_input.get("gt_sentence", []),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"loss": torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)}
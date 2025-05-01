import numpy as np
import torch
import copy

EXPECTED_KEYPOINTS = {
    'body': 9,
    'left': 21,
    'right': 21,
    'face_all': 18
}

def determine_max_frames(batch, required_parts):
    max_frames = 0
    for item in batch:
        if 'keypoints' in item[0]:
            keypoints = np.array(item[0]['keypoints'])
            max_frames = max(max_frames, keypoints.shape[0])
        else:
            for part in required_parts:
                if part in item[0] and isinstance(item[0][part], torch.Tensor):
                    tensor = item[0][part]
                    if tensor.dim() >= 3:
                        if tensor.dim() == 5:
                            frame_count = tensor.shape[2]
                        elif tensor.dim() == 4 and tensor.shape[0] == 1:
                            frame_count = tensor.shape[1]
                        else:
                            frame_count = tensor.shape[0]
                        max_frames = max(max_frames, frame_count)
    return max_frames if max_frames > 0 else 1

def get_effective_batch_size(src_input, required_parts, filtered_batch):
    for part in required_parts:
        if part in src_input and isinstance(src_input[part], torch.Tensor):
            return src_input[part].size(0)
    return len(filtered_batch)

def prepare_tensor(tensor):
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    
    if tensor.dim() == 5:
        tensor = tensor.squeeze(1).squeeze(0)
    elif tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    return tensor

def pad_with_last(data, length):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
        
    if len(data) == 0:
        return torch.zeros((length, *data.shape[1:]))
        
    if len(data) >= length:
        return torch.tensor(data[:length])
        
    padding = np.broadcast_to(data[-1], (length - len(data), *data.shape[1:]))
    return torch.tensor(np.concatenate((data, padding), axis=0))

def load_part_kp(skeletons, confs, force_ok=True):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0] if len(skeleton.shape) > 2 else skeleton
            conf = conf[0] if len(conf.shape) > 1 else conf
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + list(range(3, 11)), :]
                confidence = conf[[0] + list(range(3, 11))]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                face_indices = [i for i in list(range(23,23+17))[::2]] + list(range(83, 83+8)) + [53]
                hand_kp2d = skeleton[face_indices, :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[face_indices]
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
        
        if not kps:
            continue
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            try:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[..., None]], axis=-1), thr)
            except:
                result = np.zeros((len(kps), kps.shape[1], 3))
                scale = 1
        else:
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            if scale == 0:
                scale = 1
            result[..., :2] = result[..., :2] / scale
            result = np.clip(result, -1, 1)
            result[result[..., 2] <= thr] = 0
        
        kps_with_scores[part] = torch.tensor(result)
    
    return kps_with_scores

def crop_scale(motion, thr):
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 1, None
        
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    
    ratio = 1
    scale = max(xmax - xmin, ymax - ymin) * ratio
    
    if scale == 0 or scale is None:
        scale = 1
        
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    
    return result, scale, [xs, ys]

def process_pose_parts(batch, required_parts, max_frames):
    result = {}
    batch_items = []
    
    for item_idx, (src_dict, tgt_dict) in enumerate(batch):
        try:
            if 'keypoints' in src_dict and 'scores' in src_dict:
                processed_parts = load_part_kp(src_dict['keypoints'], src_dict['scores'], force_ok=True)
                batch_items.append((item_idx, processed_parts))
            elif any(part in src_dict for part in required_parts):
                batch_parts = {}
                for part in required_parts:
                    if part in src_dict and isinstance(src_dict[part], torch.Tensor):
                        batch_parts[part] = src_dict[part]
                if batch_parts:
                    batch_items.append((item_idx, batch_parts))
        except:
            continue
    
    for part in required_parts:
        tensors = []
        for idx, parts_dict in batch_items:
            if part in parts_dict:
                tensor = parts_dict[part]
                if tensor.dim() == 2 and tensor.shape[0] == EXPECTED_KEYPOINTS[part]:
                    tensor = tensor.unsqueeze(0)
                if tensor.shape[0] < max_frames:
                    tensor = pad_with_last(tensor, max_frames)
                else:
                    tensor = tensor[:max_frames]
                tensors.append(tensor)
        
        if tensors:
            try:
                result[part] = torch.stack(tensors)
            except:
                pass
    
    if not result:
        return {}, []
        
    filtered_batch = [batch[idx] for idx, _ in batch_items]
    return result, filtered_batch

def handle_attention_mask(batch, src_input, max_frames, batch_size):
    if not src_input or batch_size == 0:
        src_input['attention_mask'] = torch.ones((1, 1), dtype=torch.float32)
        return src_input
        
    device = None
    for v in src_input.values():
        if isinstance(v, torch.Tensor):
            device = v.device
            break
            
    src_input['attention_mask'] = torch.ones((batch_size, max_frames), dtype=torch.float32)
    if device:
        src_input['attention_mask'] = src_input['attention_mask'].to(device)
    
    return src_input

def process_target_data(batch):
    tgt_input = {}
    if not batch:
        return tgt_input
    
    sentences = []
    for item in batch:
        if len(item) > 1:
            if isinstance(item[1], dict) and 'gt_sentence' in item[1]:
                sentence = item[1]['gt_sentence']
                if isinstance(sentence, (list, tuple)):
                    sentences.extend(sentence)
                else:
                    sentences.append(sentence)
            elif isinstance(item[1], str):
                sentences.append(item[1])
    
    if sentences:
        tgt_input['gt_sentence'] = sentences
    
    return tgt_input
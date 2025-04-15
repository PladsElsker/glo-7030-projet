from DWPose.ControlNet_v1_1_nightly.annotator.dwpose.wholebody import inference_detector, inference_pose
import numpy as np
import cv2
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import pickle
import os
import onnxruntime as ort
import sys


log_file = open(f'dwpose_logs_{sys.argv[1]}.txt', 'w')
sys.stdout = log_file
sys.stderr = log_file

openasl_labels = './OpenASL/data/openasl-v1.0.tsv'
openasl_dataset = '/mnt/data/plads/open-asl/'
openasl_dataset_json_poses_path = '/mnt/data/plads/open-asl-dwpose/'


def main():
    pkl_encode_open_asl()
    # validate_open_asl_dataset()
    # encode_csl_daily_vid()
    # example_pkl_encode()


def encode_csl_daily_vid():
    video_path = 'Common-Concerns_20210925_44612-44912_16547.mp4'
    dwpose = DWPose()
    label = {
        'yid': 'pewpew',
        'start': '00:00:00.000', 
        'end': '00:00:10.000',
        'raw-text': '本来呢我们看到美国呢呃向澳大利亚出售这个核潜艇啊，是想害人，因为它是要搞大国对抗的。',
        'tokenized-text': '本来呢我们看到美国呢呃向澳大利亚出售这个核潜艇啊，是想害人，因为它是要搞大国对抗的。',
        'split': 'train',
    }

    video_poses = poses_from_video(dwpose, video_path, label)
    with open(video_poses.file_stem + '.pkl', "wb") as f:
        pickle.dump(video_poses.__dict__, f)


def validate_open_asl_dataset():
    labels = pd.read_csv(openasl_labels, sep='\t')
    video_paths = [openasl_dataset + yid + '.mp4' for yid in labels['yid'].unique()]
    for video_path in tqdm(video_paths):
        if not os.path.exists(video_path):
            print(f'Missing {video_path}')


def example_pkl_encode():
    labels = pd.read_csv(openasl_labels, sep='\t')
    dwpose = DWPose()

    index = 0
    label = {
        k: v[index]
        for k, v in labels.items()
    }

    video_path = openasl_dataset + label['yid'] + '.mp4'
    video_poses = poses_from_video(dwpose, video_path, label)
    with open(video_poses.file_stem + '.pkl', "wb") as f:
        pickle.dump(video_poses.__dict__, f)


def pkl_encode_open_asl():
    device_id = int(sys.argv[1])
    devices = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    device = devices[device_id]
    num_workers = len(devices)
    labels = pd.read_csv(openasl_labels, sep='\t')

    indices = list(range(len(labels['vid'])))
    chunks = [indices[i::num_workers] for i in range(num_workers)]
    chunk = chunks[device]

    with tqdm(desc='Kepoints extract', total=len(chunk)) as pbar:
        dwpose = DWPose(device)
        pkl_encode_open_asl_thread(chunk, dwpose, labels, pbar)


def pkl_encode_open_asl_thread(chunk, dwpose, labels, pbar=None):
    for i in chunk:
        label = {
            k: v[i]
            for k, v in labels.items()
        }

        output_path = openasl_dataset_json_poses_path + encode_filename(label['yid'], label['start'], label['end']) + '.pkl'
        if os.path.exists(output_path):
            pbar.update(1)
            continue

        video_path = openasl_dataset + label['yid'] + '.mp4'
        video_poses = poses_from_video(dwpose, video_path, label, use_tqdm=False)

        with open(output_path, "wb") as f:
            pickle.dump(video_poses.__dict__, f)
        
        if pbar is not None:
            pbar.update(1)


@dataclass
class VideoPoses:
    poses: list
    video_frames_per_second: float
    video_duration: float
    video_frame_count: int
    video_width: int
    video_height: int
    file_stem: str
    raw_text: str
    tokenized_text: str
    split: str
    yid: str
    start: str
    end: str


class DWPose():
    def __init__(self, gpu_id=None):
        providers = [('CUDAExecutionProvider', {'device_id': gpu_id})] if gpu_id is not None else ['CUDAExecutionProvider']
        onnx_det = 'DWPose/ControlNet_v1_1_nightly/annotator/ckpts/yolox_l.onnx'
        onnx_pose = 'DWPose/ControlNet_v1_1_nightly/annotator/ckpts/dw-ll_ucoco_384.onnx'

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 0
        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, sess_options=session_options, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, sess_options=session_options, providers=providers)

    def __call__(self, image):
        det_result = inference_detector(self.session_det, image)
        keypoints, scores = inference_pose(self.session_pose, det_result, image)
        return [keypoints[0]], [scores[0]]


def poses_from_video(dwpose, video_path, datapoint, use_tqdm=True):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    keypoints = []
    scores = []

    # resize_factor = 224 / min(frame_width, frame_height)
    # new_video_size = int(resize_factor * frame_width), int(resize_factor * frame_height)

    file_stem = encode_filename(datapoint['yid'], datapoint['start'], datapoint['end'])

    frame_start = timstampt_to_frame_index(datapoint['start'], fps)
    iteration_frame_count = min(timstampt_to_frame_index(datapoint['end'], fps), frame_count) - frame_start

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    it = range(iteration_frame_count)
    if use_tqdm:
        it = tqdm(it, desc=video_path)

    for _ in it:
        ret, frame = video.read()
        if not ret:
            break

        # frame = cv2.resize(frame, new_video_size)

        ks, ss = dwpose(frame)
        if ks is not None:
            keypoints.append(ks)
        else:
            keypoints.append([keypoints[-1]])

        if ss is not None:
            scores.append(ss)
        else:
            scores.append([scores[-1]])

    keypoints = np.array(keypoints)
    scores = np.array(scores)
    poses = dict(keypoints=keypoints, scores=scores)
    return VideoPoses(
        poses=poses,
        video_frames_per_second=fps,
        video_duration=iteration_frame_count / fps if fps > 0 else 0,
        video_frame_count=iteration_frame_count,
        video_width=frame_width,
        video_height=frame_height,
        file_stem=file_stem,
        raw_text=datapoint['raw-text'],
        tokenized_text=datapoint['tokenized-text'],
        split=datapoint['split'],
        yid=datapoint['yid'],
        start=datapoint['start'],
        end=datapoint['end'],
    )


def encode_filename(video_id, start_ts, end_ts):
    def safe(t): return t.replace(":", "-").replace(".", "-")
    return f"{video_id}__{safe(start_ts)}__{safe(end_ts)}"


def decode_filename(s):
    parts = s.split("__")
    if len(parts) != 3:
        raise ValueError("Invalid format")
    video_id = parts[0]
    unsafe = lambda t: t.replace("-", ":", 2).replace("-", ".")
    start = unsafe(parts[1])
    end = unsafe(parts[2])
    return video_id, start, end


def timstampt_to_frame_index(s, fps):
    hours, minutes, sec = s.split(':')
    seconds = int(hours) * 3600 + int(minutes) * 60 + float(sec)
    frame_index = int(seconds * fps)
    return frame_index


if __name__ == '__main__':
    main()

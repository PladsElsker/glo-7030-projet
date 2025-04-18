import re
from datetime import timedelta
import numpy as np


def get_lines():
    for i in range(15):
        with open(f'dwpose_logs_{i}.txt', 'r') as f:
            yield f.readlines()[-1]


def parse_eta(line):
    match = re.search(r'<(?:(\d{1,3}):)?(\d{2}):(\d{2}),', line)
    if match:
        h, m, s = match.groups()
        h = int(h) if h is not None else 0
        return timedelta(hours=h, minutes=int(m), seconds=int(s)).total_seconds()
    return -1


etas = [parse_eta(line) for line in get_lines() if parse_eta(line) > 0]
max_eta_seconds = max(etas)
min_eta_seconds = min(etas)
avg_eta_seconds = sum(etas) / len(etas)
file_id_max_eta = np.argmax([parse_eta(line) for line in get_lines()])
file_id_min_eta = np.argmin([parse_eta(line) for line in get_lines()])
max_eta = str(timedelta(seconds=int(max_eta_seconds)))
min_eta = str(timedelta(seconds=int(min_eta_seconds)))
avg_eta = str(timedelta(seconds=int(avg_eta_seconds)))


print(f"{'Maximum ETA':<20}\t | dwpose_logs_{file_id_max_eta}.txt \t | {max_eta}")
print(f"{'Minimum ETA':<20}\t | dwpose_logs_{file_id_min_eta}.txt \t | {min_eta}")
print(f"{'Average ETA':<20}\t | \t\t\t | {avg_eta}")

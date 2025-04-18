import contextlib
import sys
from pathlib import Path
sys.path[0:0] = ["uni_sign"]

import torch

from core.models.uni_sign import UniSign, UniSignModelParameters, preprocess_data
from uni_sign import config
import tempfile
import pickle



"""
This file should be removed and only exists for testing purposes
"""


def main() -> None:
    device = "cuda"
    model = UniSign(UniSignModelParameters())
    state_dict = torch.load(config.pretrained_weights["Stage2"], weights_only=True)
    model.load_state_dict(state_dict["model"], strict=False)
    model.to("cuda")

    with open('pewpew__00-00-00-000__00-00-10-000.pkl', 'rb') as dp_f:
        datapoint = pickle.load(dp_f)

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=".pkl") as f:
        pickle.dump(datapoint['poses'], f)
        temp_path = Path(f.name)

    x = preprocess_data(
        temp_path,
        datapoint['raw_text'],  # noqa: RUF001
    )
    print(datapoint['raw_text'])

    for input_data in x:
        for k, v in input_data.items():
            try:
                input_data[k] = v.to(device).to(torch.float32)
            except AttributeError as e:
                contextlib.suppress(e)

    y_hat = model(*x)
    out = model.generate(y_hat, max_new_tokens=100, num_beams=4)
    text_prediction = model.mt5_tokenizer.decode(out[0], skip_special_tokens=True)
    print(text_prediction)  # noqa: T201


def dicts_same_structure(a, b):
    if a.keys() != b.keys():
        return False, 'keys'
    for k in a:
        v1, v2 = a[k], b[k]
        if isinstance(v1, dict) and isinstance(v2, dict):
            ok, reason = dicts_same_structure(v1, v2)
            if not ok:
                return False, reason
        elif isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            if v1.shape != v2.shape:
                return False, 'shapes'
        else:
            if type(v1) != type(v2):
                return False, 'types'
    return True, None


if __name__ == "__main__":
    main()
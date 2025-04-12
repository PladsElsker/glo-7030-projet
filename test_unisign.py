import contextlib
import sys
from pathlib import Path

import torch

from core.models.uni_sign import UniSign, UniSignModelParameters, preprocess_data
from uni_sign import config

sys.path[0:0] = ["uni_sign"]

"""
This file should be removed and only exists for testing purposes
"""


def main() -> None:
    device = "cuda"
    model = UniSign(UniSignModelParameters())
    state_dict = torch.load(config.pretrained_weights["Stage2"], weights_only=True)
    model.load_state_dict(state_dict["model"], strict=False)
    model.to("cuda")
    x = preprocess_data(
        Path("/mnt/data/plads/csl-news/rgb_format/Common-Concerns_20210925_44612-44912_16547.pkl"),
        "本来呢我们看到美国呢呃向澳大利亚出售这个核潜艇啊，是想害人，因为它是要搞大国对抗的。",  # noqa: RUF001
    )

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


if __name__ == "__main__":
    main()

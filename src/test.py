import argparse

import cv2
import numpy as np
import torch

from constants import *
from network import UpresNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("out")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IN_SIZE[::-1])

    model = UpresNet()
    model.load_state_dict(torch.load(args.model))
    x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    y = model(x)

    out = y[0].permute(1, 2, 0).detach().numpy()
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out, out)


if __name__ == "__main__":
    main()

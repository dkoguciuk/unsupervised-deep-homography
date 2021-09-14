import torch
import argparse
import numpy as np
from tqdm import tqdm

from train import HomographyModel
from dataset import SyntheticDataset


@torch.no_grad()
def main(args):
    model = HomographyModel.load_from_checkpoint(args.checkpoint)
    model.eval()
    test_set = SyntheticDataset(args.test_path, rho=args.rho, filetype=args.filetype)

    mace = []
    for i in tqdm(range(args.n), total=args.n):
        img_a, patch_a, patch_b, corners, delta = test_set[i]
        patch_a = patch_a.unsqueeze(0)
        patch_b = patch_b.unsqueeze(0)
        delta_hat = model(patch_a, patch_b)
        mace.append(np.abs(delta.detach().cpu().numpy().reshape(-1) - delta_hat.detach().cpu().numpy().reshape(-1)))
    print('MACE:', np.mean(mace))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pretrained_coco.ckpt")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--rho", type=int, default=20, help="amount to perturb corners")
    parser.add_argument("--n", type=int, default=5, help="number of images to test")
    parser.add_argument("--filetype", default=".jpg")
    parser.add_argument("test_path", help="path to test images")
    args = parser.parse_args()
    main(args)

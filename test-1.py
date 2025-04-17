import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from torch.nn.functional import mse_loss

from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str,
        default="./example/remove-anything/dog.jpg",
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str,
        default="key_in", choices=["click", "key_in"],
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+',
        default=[200, 450],
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+',
        default=1,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=15,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./results",
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str,
        default="./pretrained_models/sam_vit_h_4b8939.pth",
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default="./pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )

def clamp(tensor: torch.tensor, lower_limit, upper_limit):
    return torch.max(torch.min(tensor, upper_limit), lower_limit)

def tensor_to_array(tensor: torch.Tensor):
    result = tensor.detach().cpu().numpy()
    result = np.clip(result * 255, 0, 255).astype('uint8')
    return result

def array_to_tensor(array: np.ndarray):
    return torch.from_numpy(array).float().div(255.)

if __name__ == "__main__":
    """Example usage:
    python remove_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --coords_type key_in \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean = (0.0, 0.0, 0.0)
    std = (1.0, 1.0, 1.0)
    mu = torch.tensor(mean).view(3, 1, 1).cpu()
    std = torch.tensor(std).view(3, 1, 1).cpu()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    epsilon = (8 / 255.)
    start_epsilon = (8 / 255.) / std
    step_alpha = (2 / 255.)

    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    img = load_img_to_array(args.input_img)
    reference_img = load_img_to_array("./example/remove-anything/dog_reference.png")
    dog_mask = load_img_to_array("./example/remove-anything/dog_mask.png").astype(np.uint8) * 255

    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_p = out_dir / f"mask_0.png"

    delta2 = torch.zeros(img.shape, device=device)
    delta2.requires_grad = True

    img_tensor = array_to_tensor(img)
    reference_tensor = array_to_tensor(reference_img)
    mask_tensor = torch.from_numpy(dog_mask).float()

    for i in range(5):
        attempt_input_tensor = torch.clamp(img_tensor + delta2, min=0, max=1)

        inpainted_tensor = inpaint_img_with_lama(
            attempt_input_tensor,
            mask_tensor, args.lama_config, args.lama_ckpt, device=device)
        loss = mse_loss(reference_tensor, inpainted_tensor)
        loss.backward()
        grad = delta2.grad.detach()
        d = delta2
        d = torch.clamp(d + step_alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta2.data = d
        delta2.grad.zero_()

    input = torch.clamp(img_tensor + delta2, min=0, max=1)
    save_array_to_img(tensor_to_array(input), out_dir / f"input_adversarial.png")
    save_array_to_img(img, out_dir / f"input_normal.png")
    img_inpainted = inpaint_img_with_lama(
        input,
        mask_tensor, args.lama_config, args.lama_ckpt, device=device)
    save_array_to_img(tensor_to_array(img_inpainted), out_dir / f"output_adversarial.png")

    img_inpainted = inpaint_img_with_lama(
        img_tensor,
        mask_tensor, args.lama_config, args.lama_ckpt, device=device)
    save_array_to_img(tensor_to_array(img_inpainted), out_dir / f"output_normal.png")

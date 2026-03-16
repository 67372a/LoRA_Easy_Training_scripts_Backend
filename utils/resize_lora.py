# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo

# Modified from kohya's resize script to allow removing of conv or linear dims
import os
from pathlib import Path

os.chdir("sd_scripts")

import argparse
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
import numpy as np

from library import train_util, model_util
from library.utils import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

MIN_SV = 1e-6

LORA_DOWN_UP_FORMATS = [
    ("lora_down", "lora_up"),
    ("lora_A", "lora_B"),
    ("down", "up"),
]


# Model save and load functions
def load_state_dict(file_name, dtype):
    if model_util.is_safetensors(file_name):
        sd = load_file(file_name)
        with safe_open(file_name, framework="pt") as f:
            metadata = f.metadata()
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = None

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict, dtype, metadata):
    if dtype is not None:
        for key in list(state_dict.keys()):
            if (
                type(state_dict[key]) == torch.Tensor
                and state_dict[key].dtype.is_floating_point
                and state_dict[key].dtype != dtype
            ):
                state_dict[key] = state_dict[key].to(dtype)

    if model_util.is_safetensors(file_name):
        save_file(state_dict, file_name, metadata)
    else:
        torch.save(state_dict, file_name)


# Indexing functions
def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_fro(S, target):
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2)) + 1
    index = max(1, min(index, len(S) - 1))

    return index


def index_sv_ratio(S, target):
    max_sv = S[0]
    min_sv = max_sv / target
    index = int(torch.sum(S > min_sv).item())
    return max(1, min(index, len(S) - 1))


# Modified from Kohaku-blueleaf's extract/merge functions
def extract_conv(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size, kernel_size, _ = weight.size()
    U, S, Vh = torch.linalg.svd(weight.reshape(out_size, -1).to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(
        lora_rank, in_size, kernel_size, kernel_size
    ).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank, 1, 1).cpu()
    del U, S, Vh, weight
    return param_dict


def extract_linear(weight, lora_rank, dynamic_method, dynamic_param, device, scale=1):
    out_size, in_size = weight.size()

    U, S, Vh = torch.linalg.svd(weight.to(device))

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank]
    S = S[:lora_rank]
    U = U @ torch.diag(S)
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank).cpu()
    del U, S, Vh, weight
    return param_dict


def merge_conv(lora_down, lora_up, device):
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert (
        in_rank == out_rank and kernel_size == k_
    ), f"rank {in_rank} {out_rank} or kernel {kernel_size} {k_} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    merged = lora_up.reshape(out_size, -1) @ lora_down.reshape(in_rank, -1)
    weight = merged.reshape(out_size, in_size, kernel_size, kernel_size)
    del lora_up, lora_down
    return weight


def merge_tucker_conv(lora_down, lora_up, lora_mid, device):
    in_rank, in_size, _, _ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    mid_rank_1, mid_rank_2, _, _ = lora_mid.shape
    assert (
        in_rank == out_rank == mid_rank_1 == mid_rank_2
    ), f"rank {in_rank} {out_rank} {mid_rank_1} {mid_rank_2} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)
    lora_mid = lora_mid.to(device)

    weight = torch.einsum(
        "m n ..., i m, n j -> i j ...",
        lora_mid,
        lora_up.reshape(out_size, out_rank),
        lora_down.reshape(in_rank, in_size),
    )
    del lora_up, lora_down, lora_mid
    return weight


def merge_linear(lora_down, lora_up, device):
    in_rank, in_size = lora_down.shape
    out_size, out_rank = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"

    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    weight = lora_up @ lora_down
    del lora_up, lora_down
    return weight


def get_lora_down_up_names(key):
    key_parts = key.split(".")
    for down_name, up_name in LORA_DOWN_UP_FORMATS:
        if len(key_parts) >= 2 and down_name == key_parts[-2]:
            return ".".join(key_parts[:-2]), f".{down_name}", f".{up_name}", f".{key_parts[-1]}"
        if len(key_parts) >= 1 and down_name == key_parts[-1]:
            return ".".join(key_parts[:-1]), f".{down_name}", f".{up_name}", ""
    return None, None, None, None


def delete_block_weights(state_dict, block_name, lora_down_name, lora_up_name, weight_name):
    keys = [
        block_name + lora_down_name + weight_name,
        block_name + lora_up_name + weight_name,
        block_name + ".alpha",
        block_name + ".dora_scale",
        block_name + ".lora_mid.weight",
    ]
    for key in keys:
        if key in state_dict:
            del state_dict[key]


# Calculate new rank
def rank_resize(S, rank, dynamic_method, dynamic_param, scale=1):
    if dynamic_method == "sv_ratio":
        # Calculate new dim and alpha based off ratio
        new_rank = index_sv_ratio(S, dynamic_param) + 1
    elif dynamic_method == "sv_cumulative":
        # Calculate new dim and alpha based off cumulative sum
        new_rank = index_sv_cumulative(S, dynamic_param) + 1
    elif dynamic_method == "sv_fro":
        # Calculate new dim and alpha based off sqrt sum of squares
        new_rank = index_sv_fro(S, dynamic_param) + 1
    else:
        new_rank = rank
    new_alpha = float(scale * new_rank)

    if S[0] <= MIN_SV:  # Zero matrix, set dim to 1
        new_rank = 1
        new_alpha = float(scale * new_rank)
    elif new_rank > rank:  # cap max rank at rank
        new_rank = rank
        new_alpha = float(scale * new_rank)

    # Calculate resize info
    s_sum = torch.sum(torch.abs(S))
    s_rank = torch.sum(torch.abs(S[:new_rank]))

    S_squared = S.pow(2)
    s_fro = torch.sqrt(torch.sum(S_squared))
    s_red_fro = torch.sqrt(torch.sum(S_squared[:new_rank]))
    fro_percent = float(s_red_fro / s_fro)

    return {
        "new_rank": new_rank,
        "new_alpha": new_alpha,
        "sum_retained": s_rank / s_sum,
        "fro_retained": fro_percent,
        "max_ratio": S[0] / S[new_rank - 1],
    }


def resize_lora_model(
    lora_sd,
    new_rank,
    new_conv_rank,
    save_dtype,
    device,
    dynamic_method,
    dynamic_param,
    verbose,
    del_linear,
    del_conv,
):  # sourcery skip: use-fstring-for-concatenation
    max_old_rank = None
    verbose_str = "\n"
    fro_list = []

    if dynamic_method:
        logger.info(
            f"Dynamically determining new alphas and dims based off {dynamic_method}: {dynamic_param}, max rank is {new_rank}"
        )

    lora_down_weight = None
    lora_up_weight = None

    o_lora_sd = lora_sd.copy()
    block_down_name = None
    block_up_name = None

    new_alpha = 0.0

    with torch.no_grad():
        for key, value in tqdm(lora_sd.items()):
            block_down_name, lora_down_name, lora_up_name, weight_name = get_lora_down_up_names(key)
            if block_down_name is None:
                continue

            lora_down_weight = value
            block_up_name = block_down_name
            lora_up_weight = lora_sd.get(block_up_name + lora_up_name + weight_name, None)
            lora_alpha = lora_sd.get(block_down_name + ".alpha", None)
            lora_mid_weight = lora_sd.get(block_down_name + ".lora_mid.weight", None)

            weights_loaded = lora_down_weight is not None and lora_up_weight is not None

            if weights_loaded:
                conv2d = len(lora_down_weight.size()) == 4
                old_rank = lora_down_weight.size()[0]
                max_old_rank = max(max_old_rank or 0, old_rank)
                scale = (
                    1.0
                    if lora_alpha is None
                    else lora_alpha / old_rank
                )
                if conv2d:
                    if del_conv:
                        delete_block_weights(
                            o_lora_sd,
                            block_down_name,
                            lora_down_name,
                            lora_up_name,
                            weight_name,
                        )
                        block_down_name = None
                        block_up_name = None
                        lora_down_weight = None
                        lora_up_weight = None
                        continue
                    if (
                        lora_mid_weight is not None
                        and lora_down_name == ".lora_down"
                        and lora_up_name == ".lora_up"
                        and weight_name == ".weight"
                    ):
                        full_weight_matrix = merge_tucker_conv(
                            lora_down_weight, lora_up_weight, lora_mid_weight, device
                        )
                    else:
                        full_weight_matrix = merge_conv(
                            lora_down_weight, lora_up_weight, device
                        )
                    param_dict = extract_conv(
                        full_weight_matrix,
                        new_conv_rank,
                        dynamic_method,
                        dynamic_param,
                        device,
                        scale,
                    )
                else:
                    if del_linear:
                        delete_block_weights(
                            o_lora_sd,
                            block_down_name,
                            lora_down_name,
                            lora_up_name,
                            weight_name,
                        )
                        block_down_name = None
                        block_up_name = None
                        lora_down_weight = None
                        lora_up_weight = None
                        continue
                    full_weight_matrix = merge_linear(
                        lora_down_weight, lora_up_weight, device
                    )
                    param_dict = extract_linear(
                        full_weight_matrix,
                        new_rank,
                        dynamic_method,
                        dynamic_param,
                        device,
                        scale,
                    )

                if verbose:
                    max_ratio = param_dict["max_ratio"]
                    sum_retained = param_dict["sum_retained"]
                    fro_retained = param_dict["fro_retained"]
                    if not np.isnan(fro_retained):
                        fro_list.append(float(fro_retained))

                    verbose_str += f"{block_down_name:75} | "
                    verbose_str += f"sum(S) retained: {sum_retained:.1%}, fro retained: {fro_retained:.1%}, max(S) ratio: {max_ratio:0.1f}"

                if verbose and dynamic_method:
                    verbose_str += f", dynamic | dim: {param_dict['new_rank']}, alpha: {param_dict['new_alpha']}\n"
                else:
                    verbose_str += "\n"

                new_alpha = param_dict["new_alpha"]
                o_lora_sd[block_down_name + lora_down_name + weight_name] = (
                    param_dict["lora_down"].to(save_dtype).contiguous()
                )
                o_lora_sd[block_up_name + lora_up_name + weight_name] = (
                    param_dict["lora_up"].to(save_dtype).contiguous()
                )
                o_lora_sd[block_down_name + ".alpha"] = torch.tensor(
                    param_dict["new_alpha"]
                ).to(save_dtype)
                if block_down_name + ".lora_mid.weight" in o_lora_sd:
                    del o_lora_sd[block_down_name + ".lora_mid.weight"]

                block_down_name = None
                block_up_name = None
                lora_down_weight = None
                lora_up_weight = None
                del param_dict

    if verbose:
        print(verbose_str)
        print(
            f"Average Frobenius norm retention: {np.mean(fro_list):.2%} | std: {np.std(fro_list):0.3f}"
        )
    logger.info("resizing complete")
    return o_lora_sd, max_old_rank, new_alpha


def resize(args):
    if args.save_to is None or not (
        args.save_to.endswith(".ckpt")
        or args.save_to.endswith(".pt")
        or args.save_to.endswith(".pth")
        or args.save_to.endswith(".safetensors")
    ):
        raise Exception(
            "The --save_to argument must be specified and must be a .ckpt , .pt, .pth or .safetensors file."
        )

    args.new_conv_rank = (
        args.new_conv_rank if args.new_conv_rank is not None else args.new_rank
    )

    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        return torch.bfloat16 if p == "bf16" else None

    if args.dynamic_method and not args.dynamic_param:
        raise Exception("If using dynamic_method, then dynamic_param is required")

    merge_dtype = str_to_dtype(
        "float"
    )  # matmul method above only seems to work in float32
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    logger.info("loading Model...")
    lora_sd, metadata = load_state_dict(args.model, merge_dtype)

    logger.info("Resizing Lora...")
    state_dict, old_dim, new_alpha = resize_lora_model(
        lora_sd,
        args.new_rank,
        args.new_conv_rank,
        save_dtype,
        args.device,
        args.dynamic_method,
        args.dynamic_param,
        args.verbose,
        args.del_linear,
        args.del_conv,
    )

    # update metadata
    if metadata is None:
        metadata = {}

    comment = metadata.get("ss_training_comment", "")

    if not args.dynamic_method:
        if args.del_conv:
            conv_desc = "(conv: Deleted Conv Dims)"
        else:
            conv_desc = (
                ""
                if args.new_rank == args.new_conv_rank
                else f" (conv: {args.new_conv_rank})"
            )
        if args.del_linear:
            metadata["ss_training_comment"] = (
                f"Deleted Linear Dims{conv_desc}; {comment}"
            )
        else:
            metadata["ss_training_comment"] = (
                f"dimension is resized from {old_dim} to {args.new_rank}{conv_desc}; {comment}"
            )
        metadata["ss_network_dim"] = str(0 if args.del_linear else args.new_rank)
        metadata["ss_network_alpha"] = str(new_alpha)
    else:
        if args.del_linear:
            linear_message = "Deleted Linear Dims"
        else:
            linear_message = f"Dynamically resized linear dims with {args.dynamic_method}: {args.dynamic_param} from {old_dim}"
        if args.del_conv:
            conv_message = "Deleted Conv Dims"
        else:
            conv_message = f"Dynamically resized conv dims with {args.dynamic_method}: {args.dynamic_param}"

        metadata["ss_training_comment"] = f"{linear_message}({conv_message}); {comment}"
        metadata["ss_network_dim"] = "Dynamic"
        metadata["ss_network_alpha"] = "Dynamic"

    for key in list(state_dict.keys()):
        value = state_dict[key]
        if (
            type(value) == torch.Tensor
            and value.dtype.is_floating_point
            and value.dtype != save_dtype
        ):
            state_dict[key] = value.to(save_dtype)

    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(
        state_dict, metadata
    )
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    logger.info(f"saving model to: {args.save_to}")
    save_to_file(args.save_to, state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, float if omitted / 保存時の精度、未指定時はfloat",
    )
    parser.add_argument(
        "--new_rank",
        type=int,
        default=4,
        help="Specify rank of output LoRA / 出力するLoRAのrank (dim)",
    )
    parser.add_argument(
        "--new_conv_rank",
        type=int,
        default=None,
        help="Specify rank of output LoRA for Conv2d 3x3, None for same as new_rank / 出力するConv2D 3x3 LoRAのrank (dim)、Noneでnew_rankと同じ",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LoRA model to resize at to new rank: ckpt or safetensors file / 読み込むLoRAモデル、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display verbose resizing information / rank変更時の詳細情報を出力する",
    )
    parser.add_argument(
        "--dynamic_method",
        type=str,
        default=None,
        choices=[None, "sv_ratio", "sv_fro", "sv_cumulative"],
        help="Specify dynamic resizing method, --new_rank is used as a hard limit for max rank",
    )
    parser.add_argument(
        "--dynamic_param",
        type=float,
        default=None,
        help="Specify target for dynamic reduction",
    )
    parser.add_argument(
        "--del_conv",
        action="store_true",
        help="Removes the Conv Dims of the model while resizing",
    )
    parser.add_argument(
        "--del_linear",
        action="store_true",
        help="Removes the Linear Dims of the model while resizing",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    output_folder = Path(args.save_to).parent
    if output_folder.name == "default_output":
        output_folder = Path("../default_output")
        if not output_folder.is_dir():
            output_folder.mkdir()
        args.save_to = (
            output_folder.joinpath(Path(args.save_to).name).resolve().as_posix()
        )
    resize(args)

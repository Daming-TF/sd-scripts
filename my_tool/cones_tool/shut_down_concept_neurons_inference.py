import argparse
import os
import torch
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
import networks.lora as lora
import library.model_util as model_util
# from networks.merge_lora import merge_about_cones
from my_tool.merge_param import image_inference


def shut_down_cones(text_encoder, unet, models, ratios, merge_dtype, concept_neurons_json):
    # assert len(models) == len(ratios)
    text_encoder.to(merge_dtype)
    unet.to(merge_dtype)

    # get the concept nerous dict
    import json
    with open(concept_neurons_json, 'r') as json_file:
        concept_neurons_dict = json.load(json_file)

    # create module map
    name_to_module = {}
    for i, root_module in enumerate([text_encoder, unet]):
        if i == 0:
            prefix = lora.LoRANetwork.LORA_PREFIX_TEXT_ENCODER
            target_replace_modules = lora.LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
        else:
            prefix = lora.LoRANetwork.LORA_PREFIX_UNET
            target_replace_modules = (
                    lora.LoRANetwork.UNET_TARGET_REPLACE_MODULE + lora.LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module

    # if models is None or ratios is None:
    #     ValueError(f"models is None or ratios is None")
    #     exit(0)

    for cones_key, param_dict in concept_neurons_dict.items():
        with torch.no_grad():
            module = name_to_module[cones_key]
            module.weight.fill_(0)


def merge_param(args):
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype(args.precision)

    print(f"loading SD model: {args.sd_model}")

    # load model
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.sd_model)

    # merge param
    # shut_down_cones(text_encoder, unet, args.models, args.ratios, merge_dtype,
    #                   concept_neurons_json=args.concept_nerous_json)

    # text2image
    unet.eval()
    text_encoder.eval()
    vae.eval()
    image_inference(text_encoder, vae, unet, args)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2",
        # action="store_true",
        default=False,
        help="load Stable Diffusion v2.x model / Stable Diffusion 2.xのモデルを読み込む")
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default=r'../checkpoint/v1-5-pruned.ckpt',
        help="Stable Diffusion model to load: ckpt or safetensors file, merge LoRA models if omitted / 読み込むモデル、ckptまたはsafetensors。省略時はLoRAモデル同士をマージする",
    )
    parser.add_argument(
        "--models", type=str, nargs="*", help="LoRA models to merge: ckpt or safetensors file")
    parser.add_argument(
        "--ratios", type=float, nargs="*", help="ratios for each model")

    # add
    parser.add_argument(
        "--seed", type=int, default=int("0607105102"),
    )
    parser.add_argument(
        "--prompt_txt", type=str, default=r'../config/prompt_test.txt'
    )
    parser.add_argument(
        "--save_dir", type=str, default=r'E:\Data\test\debug'
    )
    parser.add_argument(
        "--remark_info", type=str, default=r'debug'
    )

    # cones
    parser.add_argument(
        "--concept_nerous_json", type=str,
    )

    # process blocks

    # unknow
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )

    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）",
    )

    return parser


# def tran_path(args):
#     def split_path(path):
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         dirs_list = path.split('/')
#         dirs_list.insert(0, current_dir)
#         return os.path.join(*dirs_list)
#
#     args.sd_model = split_path(args.sd_model)
#     args.save_to = split_path(args.save_to)
#     if not os.path.exists(args.save_to):
#         os.mkdir(os.path.dirname(args.save_to))
#     args.models = [split_path(path)for path in args.models]
#     args.prompt_txt = split_path(args.prompt_txt)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    # tran_path(args)

    merge_param(args)

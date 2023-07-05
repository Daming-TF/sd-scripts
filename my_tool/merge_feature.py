import argparse
import os
import torch
import sys
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
import library.model_util as model_util
# from networks.lora import create_network_from_weights
from my_tool.my_lib.merge import create_lora_from_weights
from merge_param import image_inference


def merge_feature(args):
    # load sd model
    text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(False, args.sd_model)

    # create lora network
    networks = create_lora_from_weights(1.0, args.models[0], vae, text_encoder, unet)
    networks.load_weights(args.models, args.ratios)
    # networks.apply_to(text_encoder, unet, apply_text_encoder=False, apply_unet=True)

    # text2image
    eval_prepare(unet, text_encoder, vae)
    image_inference(text_encoder, vae, unet, args)


def eval_prepare(unet, text_encoder, vae):
    unet.eval()
    text_encoder.eval()
    vae.eval()


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v2",
        # action="store_true",
        default=False,
        help="load Stable Diffusion v2.x model / Stable Diffusion 2.xのモデルを読み込む")
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
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

    # need to rewrite param
    parser.add_argument(
        "--save_to", type=str,
        default=r'D:../result/merge_test/debug.safetensors', help="destination file name: ckpt or safetensors file "
    )
    parser.add_argument(
        "--models", type=str, nargs="*", help="LoRA models to merge: ckpt or safetensors file / マージするLoRAモデル、ckptまたはsafetensors",
        default=[r'../result/healing_wo_te.safetensors']       # r'../result/healing_wo_te.safetensors',
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率",
                        default=[1])

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


if __name__ == "__main__":
    parser = setup_parser()

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
    args = parser.parse_args()
    # tran_path(args)

    merge_feature(args)

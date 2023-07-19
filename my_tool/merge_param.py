import argparse
import os
import torch
from diffusers import DDIMScheduler
import cv2
import numpy as np
import sys
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import library.model_util as model_util
import library.train_util as train_util
from networks.merge_lora import save_to_file, merge_lora_models
from networks.merge_lora import merge_to_sd_model, layerwise_merge, layerwise_merge_enhanced_version
from my_tool.prompt_preprocess import prompt_preprocess


# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


# def train_info_record(infos, save_dir):
#     txt_path = os.path.join(save_dir, 'record.txt')
#     with open(txt_path, 'a', encoding="utf-8") as file:
#         # file.write(filename + '\n')
#         for info in infos.split('@@'):
#             file.write(info.rep + '\n')


def image_process(image, info):
    image = cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # add remark info
    org = (10, 30)  # 文字的位置坐标
    font = cv2.FONT_HERSHEY_SIMPLEX  # 文字的字体
    font_scale = 1  # 文字的大小
    color = (0, 0, 255)  # 文字的颜色，这里使用红色 (BGR格式)
    thickness = 2  # 文字的线宽

    (_, text_height), _ = cv2.getTextSize('A', font, font_scale, thickness)

    infos = info.split('@')
    for i, info in enumerate(infos):
        position = (org[0], org[1]+i*text_height)
        cv2.putText(image, info, position, font, font_scale, color, thickness)

    return image


def seed_init(seed):
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def image_inference(text_encoder, vae, unet, args):
    txt_path = args.prompt_txt
    info = args.remark_info
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    def get_prompts(txt_path):
        if txt_path.endswith(".txt"):
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
        else:
            exit(1)
        return prompts

    prompts = get_prompts(txt_path)

    scheduler_cls = DDIMScheduler
    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # print("set clip_sample to True")
        scheduler.config.clip_sample = True

    tokenizer = train_util.load_tokenizer(args)

    pipeline = StableDiffusionLongPromptWeightingPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        clip_skip=args.clip_skip,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    device = torch.device(type='cuda')
    pipeline.to(device)

    # # record layer wise info
    # train_info_record(info, save_dir)

    for i, prompt_txt in enumerate(prompts):
        prompt, height, width, sample_steps, scale, negative_prompt, seed = prompt_preprocess(prompt_txt)

        seed = args.seed if seed is None else seed
        if seed is not None:
            seed_init(seed)

        height = max(64, height - height % 8)  # round to divisible by 8
        width = max(64, width - width % 8)  # round to divisible by 8
        print(f" prompt: {prompt} \n negative_prompt: {negative_prompt} \n height: {height} \n width: {width} "
              f" sample_steps: {sample_steps} \n scale: {scale}")

        image = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            guidance_scale=scale,
            negative_prompt=negative_prompt,
        ).images[0]

        # info process
        prompt = prompt.replace(', ', '@').replace(' ', '_')
        if negative_prompt is None:
            negative_prompt = ''
        else:
            negative_prompt = negative_prompt.replace(', ', '@').replace(' ', '_')

        # filename = f"{prompt}@@{negative_prompt}.png"
        filename = f"{prompt[:50]}.png"

        image = image_process(image, info)

        print(f"saving >> {os.path.join(save_dir, filename)} <<  ......")
        cv2.imwrite(os.path.join(save_dir, filename), image)


def merge_param(args):
    if args.ratios is not None and args.locked_keys is not None:
        assert len(args.locked_keys) == len(args.ratios)
    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    if args.sd_model is not None:
        print(f"loading SD model: {args.sd_model}")

        # load model
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.sd_model)

        # merge param
        if args.class_blocks_ratios is not None:
            layerwise_merge(text_encoder, unet, args.models, merge_dtype,
                            locked_keys=args.locked_keys,
                            class_blocks_ratios=args.class_blocks_ratios)
        elif args.multi_lora_ratios is not None:
            layerwise_merge_enhanced_version(text_encoder, unet, args.models, merge_dtype,
                                             locked_keys=args.locked_keys,
                                             multi_lora_ratios=args.multi_lora_ratios)
        else:
            merge_to_sd_model(text_encoder, unet, args.models, args.ratios, merge_dtype, locked_keys=args.locked_keys)

        # if args.class_blocks_index is None and args.class_blocks_ratios is None:
        #     merge_to_sd_model(text_encoder, unet, args.models, args.ratios, merge_dtype, locked_keys=args.locked_keys)
        # elif args.class_blocks_index is not None and args.class_blocks_ratios is not None:
        #     layer_wise_merge(text_encoder, unet, args.models, args.ratios, merge_dtype,
        #                      locked_keys=args.locked_keys,
        #                      # class_blocks_index=args.class_blocks_index,
        #                      class_blocks_ratios=args.class_blocks_ratios)
        # else:
        #     ValueError("Parameters '--class_blocks_index' or '--class_blocks_ratios' are incorrect")

        # save merge_checkpoint
        if args.save_to is not None:
            print(f"saving SD model to: {args.save_to}")
            model_util.save_stable_diffusion_checkpoint(args.v2, args.save_to, text_encoder, unet, args.sd_model, 0, 0, save_dtype, vae)

        # text2image
        unet.eval()
        text_encoder.eval()
        vae.eval()
        image_inference(text_encoder, vae, unet, args)

    else:
        state_dict = merge_lora_models(args.models, args.ratios, merge_dtype)

        print(f"saving model to: {args.save_to}")
        save_to_file(args.save_to, state_dict, state_dict, save_dtype)


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
    parser.add_argument(
        "--save_to", type=str, help="destination file name: ckpt or safetensors file "
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
    parser.add_argument(
        "--locked_keys", type=str, nargs='+',
        default=None, help="options:['down_blocks', 'mid_block', 'up_blocks']"
    )

    # process class model
    # layer_wise
    parser.add_argument(
        "--class_blocks_ratios", type=float, nargs='+',
        default=None
    )
    # multi lora layer-wise
    parser.add_argument(
        "--multi_lora_ratios", type=float, nargs='+',
        default=None
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

import torch
import re


def prompt_preprocess(prompt):
    if isinstance(prompt, dict):
        negative_prompt = prompt.get("negative_prompt")
        sample_steps = prompt.get("sample_steps", 30)
        width = prompt.get("width", 512)
        height = prompt.get("height", 512)
        scale = prompt.get("scale", 7.5)
        seed = prompt.get("seed")
        prompt = prompt.get("prompt")
    else:
        # prompt = prompt.strip()
        # if len(prompt) == 0 or prompt[0] == "#":
        #     continue

        # subset of gen_img_diffusers
        prompt_args = prompt.split(" --")
        prompt = prompt_args[0]
        negative_prompt = None
        sample_steps = 30
        width = height = 512
        scale = 7.5
        seed = None
        for parg in prompt_args:
            try:
                m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                if m:
                    width = int(m.group(1))
                    continue

                m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                if m:
                    height = int(m.group(1))
                    continue

                m = re.match(r"d (\d+)", parg, re.IGNORECASE)
                if m:
                    seed = int(m.group(1))
                    continue

                m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                if m:  # steps
                    sample_steps = max(1, min(1000, int(m.group(1))))
                    continue

                m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                if m:  # scale
                    scale = float(m.group(1))
                    continue

                m = re.match(r"n (.+)", parg, re.IGNORECASE)
                if m:  # negative prompt
                    negative_prompt = m.group(1)
                    continue

            except ValueError as ex:
                print(f"Exception in parsing / 解析エラー: {parg}")
                print(ex)

    return prompt, height, width, sample_steps, scale, negative_prompt, seed



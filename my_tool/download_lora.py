import requests
import os
from tqdm import tqdm
import argparse
import time

def main(args):
    save_dir = r'E:\LoRA_Data'
    txt_path = r'E:\LoRA_Data\lora_info.txt'

    url = "https://civitai.com/api/v1/models"
    limit = args.limit if args.limit else 3
    params = {
        "limit": limit,
        "types": "LORA",
        "BaseModels": "SD 1.5"
    }
    headers = {"Content-Type": "application/json"}

    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        response_datas = response.json()
        # print(response_datas)

        for response_data in tqdm(response_datas['items'][37:]):
            info = response_data['name']+'\t'
            stats = response_data['stats']
            if stats['downloadCount'] < 150:
                continue

            # get download url
            download_url = response_data["modelVersions"][0]['downloadUrl']

            # Connection retry
            max_retries = 5  # 最大重试次数
            retry_delay = 2  # 重试延迟时间（秒）
            for _ in range(max_retries):
                try:
                    model_response = requests.get(download_url, headers=headers)
                    if response.status_code == 200:
                        # 成功获取响应
                        print("请求成功")
                        break  # 跳出循环，不再重试
                    else:
                        # 处理其他响应状态码
                        print("请求失败，状态码:", response.status_code)
                except requests.exceptions.RequestException as e:
                    # 处理请求异常
                    print("请求异常:", e)

                time.sleep(retry_delay)
            else:
                print("重试次数超过最大限制，无法成功请求")

            if model_response.status_code == 200:
                content_disposition = model_response.headers.get("content-disposition")
                if content_disposition:
                    filename = content_disposition.split("filename=")[-1].strip("\"'")
                else:
                    filename = "downloaded_file"

                info += filename

                save_path = os.path.join(save_dir, filename)
                with open(save_path, "wb") as file:
                    file.write(model_response.content)
                    # for chunk in model_response.iter_content(chunk_size=8192):
                    #     if chunk:
                    #         file.write(chunk)

                with open(txt_path, "a", encoding="utf-8") as file:
                        file.write(info + "\n")

                print("文件下载完成:", filename)
            else:
                print("文件下载失败")

    else:
        print("请求失败")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int
    )
    args = parser.parse_args()
    main(args)

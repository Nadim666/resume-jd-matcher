# -*- coding: utf-8 -*-
"""下载模型文件到 model/ 目录（一次性，约 24MB）。

用到的模型：BAAI/bge-small-zh-v1.5 的 ONNX 量化版。
    来源 https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5
    （或 huggingface.co/Xorbits/bge-small-zh-v1.5）

用法：python get_model.py
"""
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

# 量化版足够用：24MB，比 fp32 的 95MB 快三倍左右，acc 差别在打分任务上看不出来
FILES = {
    "model_quantized.onnx":
        "https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5/resolve/master/model_quantized.onnx",
    "tokenizer.json":
        "https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5/resolve/master/tokenizer.json",
    "config.json":
        "https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5/resolve/master/config.json",
    "tokenizer_config.json":
        "https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5/resolve/master/tokenizer_config.json",
    "special_tokens_map.json":
        "https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5/resolve/master/special_tokens_map.json",
    "vocab.txt":
        "https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5/resolve/master/vocab.txt",
}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(MODEL_DIR, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print("skip", name)
            continue
        print("downloading", name, "...")
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as e:
            print("  失败：", e)
            print("  可以手动去 https://modelscope.cn/models/Xorbits/bge-small-zh-v1.5 下载",
                  name, "放进 model/ 目录")
            return 1
    # 量化版文件名统一成 model.onnx，代码里只认这一个名字
    q = os.path.join(MODEL_DIR, "model_quantized.onnx")
    onnx = os.path.join(MODEL_DIR, "model.onnx")
    if os.path.exists(q) and not os.path.exists(onnx):
        os.replace(q, onnx)
    print("\n完成。现在可以运行：python run_demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

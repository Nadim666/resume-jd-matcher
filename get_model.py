# -*- coding: utf-8 -*-
"""下载模型文件到 model/ 目录（一次性，约 23MB）。

模型：BAAI/bge-small-zh-v1.5 的 ONNX 量化版（int8），由 Xenova 转换。
源仓库：https://modelscope.cn/models/Xenova/bge-small-zh-v1.5
      （HuggingFace 镜像同名：huggingface.co/Xenova/bge-small-zh-v1.5）

不装 PyTorch，靠 onnxruntime 直接跑推理。

用法：python get_model.py
"""
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

# 量化版 22.9MB，比 fp32 的 90MB 快三倍，文本打分任务上看不出精度差
REPO = "Xenova/bge-small-zh-v1.5"
REV = "master"
MIRRORS = [
    "https://modelscope.cn/models/{repo}/resolve/{rev}/{path}",   # 国内可达，排第一
    "https://huggingface.co/{repo}/resolve/main/{path}",          # 备用（国内常超时）
]

FILES = {
    "model.onnx": "onnx/model_quantized.onnx",   # 本地统一叫 model.onnx，代码只认这个名字
    "tokenizer.json": "tokenizer.json",
    "config.json": "config.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "special_tokens_map.json": "special_tokens_map.json",
    "vocab.txt": "vocab.txt",
}


def download(rel_path, dst):
    """按镜像顺序试，第一个成功的返回 (True, url)。

    坑：HuggingFace 在国内多数网络下直连超时，所以 ModelScope 必须排第一；
    判定成功用「状态码 + 内容长度」，别只看文件大小 —— 5 秒超时留下的空文件
    和真的下到 0 字节长得很像，分不清就没法给用户有用的报错。
    """
    last = None
    for tpl in MIRRORS:
        url = tpl.format(repo=REPO, rev=REV, path=rel_path)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                expect = r.headers.get("Content-Length")
            if not data:
                last = "空响应：%s" % url
                continue
            if expect and len(data) != int(expect):
                last = "内容不完整(%d/%s)：%s" % (len(data), expect, url)
                continue
            with open(dst, "wb") as f:
                f.write(data)
            return True, url
        except Exception as e:
            last = "%s -> %s: %s" % (url, type(e).__name__, e)
    print("   失败：", last)
    return False, last


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    failed = []
    for local_name, rel in FILES.items():
        dst = os.path.join(MODEL_DIR, local_name)
        # 阈值别设成 1KB：config.json 只有 716B，会被当成"没下完"每次重下
        if os.path.exists(dst) and os.path.getsize(dst) > 16:
            print("skip     ", local_name)
            continue
        print("download ", local_name, "  <-", rel)
        ok, info = download(rel, dst)
        if ok:
            print("          %.1f MB" % (os.path.getsize(dst) / 1024 / 1024))
        else:
            failed.append(local_name)

    if failed:
        print("\n以下文件没下载成功：", ", ".join(failed))
        print("可以手动去 https://modelscope.cn/models/%s 的文件列表里下载，" % REPO)
        print("onnx/model_quantized.onnx 改名为 model.onnx，其余原样放进 model/ 目录。")
        return 1
    print("\n完成。现在可以运行：python run_demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

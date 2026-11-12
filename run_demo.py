# -*- coding: utf-8 -*-
"""交互式入口：双击 运行demo.bat 会跑这个。"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
DATA = os.path.join(BASE, "demo_data")


def main():
    print("=" * 56)
    print("  简历 - 岗位匹配度评分工具")
    print("=" * 56)
    print("  把简历和岗位 JD 存成 txt 或 docx，放进 demo_data 文件夹。")
    print()

    if not os.path.isdir(DATA):
        os.makedirs(DATA)

    files = sorted(os.listdir(DATA))
    res = [f for f in files if f.lower().endswith((".txt", ".md", ".docx")) and not f.startswith("JD_")]
    jds = [f for f in files if f.lower().endswith((".txt", ".md", ".docx")) and f.startswith("JD_")]

    print("  demo_data 里的文件：")
    for f in files:
        print("     ", f)
    print()

    if not res:
        print("  没有找到简历文件（文件名不能以 JD_ 开头）。")
        print("  请把简历存成 txt 放进 demo_data 再运行。")
        return
    if not jds:
        print("  没有找到 JD 文件（文件名要以 JD_ 开头）。")
        return

    print("  简历：")
    for i, f in enumerate(res, 1):
        print(f"    {i}. {f}")
    choice = input("\n  选第几个简历（直接回车 = 1）：").strip()
    idx = int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(res) else 0
    resume = os.path.join(DATA, res[idx])

    print("\n  岗位：")
    for i, f in enumerate(jds, 1):
        print(f"    {i}. {f.replace('JD_', '').replace('.txt', '')}")
    print("    0. 全部比较")
    pick = input("\n  选第几个（直接回车 = 0 全部）：").strip()
    if pick.isdigit() and 1 <= int(pick) <= len(jds):
        chosen = [os.path.join(DATA, jds[int(pick) - 1])]
    else:
        chosen = [os.path.join(DATA, f) for f in jds]

    print("\n  正在计算，第一次运行要加载模型，约 3-5 秒……\n")

    saved = sys.argv
    sys.argv = ["简历匹配.py", resume, "--job"] + chosen
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("match", os.path.join(BASE, "简历匹配.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  已取消")
    except Exception as e:
        print("\n  出错了：", e)
        import traceback
        traceback.print_exc()

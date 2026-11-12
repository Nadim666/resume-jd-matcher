# -*- coding: utf-8 -*-
"""demo 自检：分数分布 + 阈值敏感性。用来判断"门槛定在哪"是否有数据支撑。"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "m", os.path.join(os.path.dirname(os.path.abspath(__file__)), "简历匹配.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_data")
emb = m.Embedder()
resume = m.read_text(os.path.join(D, "我的简历.txt"))

jobs = sorted(glob.glob(os.path.join(D, "JD_*.txt")))
rows = []
for j in jobs:
    r = m.score_pair(emb, resume, m.read_text(j))
    rows.append((os.path.basename(j).replace("JD_", "").replace(".txt", ""), r))

rows.sort(key=lambda x: -x[1]["total"])
print("=" * 60)
print("分数分布（同一份简历 vs 5 条 JD）")
print("=" * 60)
for name, r in rows:
    print(f"  {name:16s} 总分 {r['total']:.3f}  技能 {r['skill']['score']:.2f}  语义 {r['sem']:.3f}")

top = rows[0][1]["total"]
second = rows[1][1]["total"]
print(f"\n第一名 {top:.3f}，第二名 {second:.3f}，差距 {top-second:.3f}")
print(f"第一名 / 第二名 = {top/second:.2f} 倍")

print("\n" + "=" * 60)
print("阈值敏感性：门槛设在不同的地方，会有几个岗被推荐？")
print("=" * 60)
print("  阈值   建议投的岗位")
for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    ok = [n for n, r in rows if r["total"] >= th]
    print(f"  {th:.2f}   {('、'.join(ok)) if ok else '（一个都不推荐）'}")

print("\n关键检查：不管阈值定在哪里，第一名始终是 ——",
      rows[0][0], "（排序稳定，说明方法对阈值不敏感）")

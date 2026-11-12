# -*- coding: utf-8 -*-
"""
简历-JD 匹配度评分工具（demo 版）

用法：
    python 简历匹配.py 我的简历.txt 岗位1JD.txt 岗位2JD.txt ...
    python 简历匹配.py --job 岗位JD.txt 简历1.txt 简历2.txt ...    # 一份 JD 比多份简历

支持的输入格式：.txt / .md / .docx
不联网，模型在本地 model/ 目录里。
"""
import argparse
import os
import re
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

# ---------------------------------------------------------------- 读文件

def read_text(path):
    low = path.lower()
    if low.endswith(".docx"):
        import zipfile
        from xml.etree import ElementTree as ET
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml")
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        root = ET.fromstring(xml)
        out = []
        for p in root.iter(ns + "p"):
            s = "".join(t.text or "" for t in p.iter(ns + "t"))
            if s.strip():
                out.append(s)
        return "\n".join(out)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_sentences(text, min_len=6):
    """按标点和换行切句；太短的（"任职要求"这种）丢掉，它不带语义信息。"""
    parts = re.split(r"[。；;！!？?\n•·]", text)
    out = []
    for p in parts:
        p = p.strip(" \t，,、-—:：()（）【】[]")
        # 去掉行首的 "1." "（2）" 这种编号
        p = re.sub(r"^[0-9]+[.、)）]\s*", "", p)
        p = re.sub(r"^[（(][0-9一二三四五六七八九十]+[)）]\s*", "", p)
        if len(p) >= min_len:
            out.append(p)
    return out

# ---------------------------------------------------------------- 模型

class Embedder:
    def __init__(self):
        self.tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self.tok.enable_truncation(max_length=512)
        # 坑：这个 tokenizer 的 normalizer 是 lowercase=False，但词表存的是小写，
        # 结果 "Python" 被切成 [UNK]。实测打开小写后 python/sql/excel 都正常。
        self.tok.normalizer.lowercase = True
        self.sess = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"),
                                         providers=["CPUExecutionProvider"])

    def encode(self, texts):
        if not texts:
            return np.zeros((0, 512), dtype=np.float32)
        encs = self.tok.encode_batch(texts)
        L = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), L), dtype=np.int64)
        mask = np.zeros((len(encs), L), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
        out = self.sess.run(None, {
            "input_ids": ids,
            "attention_mask": mask,
            "token_type_ids": np.zeros_like(ids),
        })[0]
        v = out[:, 0, :]                       # BGE：取 [CLS]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

# ---------------------------------------------------------------- 技能词表

SKILLS = {
    "Python":     ["python"],
    "C++":        ["c++", "cpp"],
    "Java":       ["java", "spring"],  # 注意：短词用词边界，否则 java 会命中 javascript
    "前端":       ["html", "css", "javascript", "react", "vue", "typescript"],
    "SQL":        ["sql", "mysql", "数据库"],
    "Excel/办公": ["excel", "wps", "office", "ppt", "飞书"],
    "数据标注":   ["标注", "标定"],
    "大模型":     ["大模型", "llm", "gpt", "提示词", "prompt", "agent"],
    "数据分析":   ["数据分析", "统计", "pandas", "numpy"],
    "医学":       ["医学", "临床", "医疗", "生物医学"],
    "文档能力":   ["文档", "报告", "写作", "整理", "周报"],
    "沟通协调":   ["沟通", "协调", "对接", "团队"],
    "护士资格":   ["护士执业资格", "执业资格", "护士证"],
    "算法":       ["算法", "深度学习", "机器学习", "pytorch", "tensorflow"],
}


def hit_any(text, words):
    """子串匹配，但短词(≤4字符)要求前后不是字母，避免 java 命中 javascript。"""
    low = text.lower()
    for w in words:
        if len(w) <= 4 and w.isascii():
            if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low):
                return True
        elif w in low:
            return True
    return False

# ---------------------------------------------------------------- 评分

def skill_layer(resume, jd):
    """
    抽 JD 要求（必须 / 加分），逐条对照简历。
    JD 里带"优先/加分/者优先"的句子算加分项，其余算硬要求。
    """
    jd_sents = split_sentences(jd)
    hard, soft = [], []
    for s in jd_sents:
        names = [k for k, ws in SKILLS.items() if hit_any(s, ws)]
        if not names:
            continue
        is_soft = any(w in s for w in ("优先", "加分", "更好", "有经验者"))
        for n in names:
            (soft if is_soft else hard).append(n)

    hard = list(dict.fromkeys(hard))
    soft = [s for s in dict.fromkeys(soft) if s not in hard]

    hard_ok = [k for k in hard if hit_any(resume, SKILLS[k])]
    hard_miss = [k for k in hard if k not in hard_ok]
    soft_hit = [k for k in soft if hit_any(resume, SKILLS[k])]
    soft_miss = [k for k in soft if k not in soft_hit]

    hard_rate = len(hard_ok) / len(hard) if hard else 1.0
    soft_rate = len(soft_hit) / len(soft) if soft else 0.0
    score = 0.7 * hard_rate + 0.3 * soft_rate
    if hard_miss:                       # 硬要求缺一条就封顶
        score = min(score, 0.3)
    return {
        "score": score, "hard_ok": hard_ok, "hard_miss": hard_miss,
        "soft_hit": soft_hit, "soft_miss": soft_miss,
    }


def semantic_layer(emb, resume, jd):
    rs, js = split_sentences(resume), split_sentences(jd)
    if not rs or not js:
        return 0.0
    R, J = emb.encode(rs), emb.encode(js)
    sim = R @ J.T
    return float(sim.mean())


# 这些是"条件/态度/时间"类的话，不是能力要求，不列进"没覆盖"
NON_SKILL = ("到岗", "每周", "实习 3 个月", "三个月", "应届", "薪", "地点", "学历要求",
             "细心", "耐心", "接受重复", "吃苦", "抗压", "责任")


def coverage(emb, resume, jd, th=0.62):
    """逐条 JD 要求，找简历里最像的那句；相似度低说明简历没真正覆盖这条。"""
    rs, js = split_sentences(resume), split_sentences(jd)
    if not rs or not js:
        return []
    R, J = emb.encode(rs), emb.encode(js)
    sim = R @ J.T
    rows = []
    for j, s in enumerate(js):
        if any(w in s for w in NON_SKILL):
            continue                       # 时间/态度类要求跳过
        if s.endswith("实习生") or s.endswith("岗位"):
            continue                       # 这是岗位名称，不是要求
        i = int(sim[:, j].argmax())
        v = float(sim[i, j])
        rows.append({"jd": s, "resume": rs[i], "sim": v, "covered": v >= th})
    return rows


def score_pair(emb, resume, jd, w_skill=0.6):
    sk = skill_layer(resume, jd)
    sem = semantic_layer(emb, resume, jd)
    total = w_skill * sk["score"] + (1 - w_skill) * sem
    if sk["hard_miss"]:
        total = min(total, 0.55)        # 硬要求缺失时不允许进"建议投"
    verdict = "建议投" if total >= 0.55 and not sk["hard_miss"] else ("可以投，但先补" if total >= 0.40 else "不建议投")
    return {"total": total, "sem": sem, "skill": sk, "verdict": verdict,
            "cov": coverage(emb, resume, jd)}

# ---------------------------------------------------------------- 输出

def bar(v, n=20):
    k = int(round(v * n))
    return "█" * k + "·" * (n - k)


def main():
    ap = argparse.ArgumentParser(description="简历-JD 匹配度评分")
    ap.add_argument("resume", nargs="+", help="简历文件（可多个）")
    ap.add_argument("--job", nargs="+", required=True, help="JD 文件（可多个）")
    ap.add_argument("--detail", action="store_true", help="逐条显示 JD 要求覆盖情况")
    args = ap.parse_args()

    emb = Embedder()
    resumes = [(os.path.basename(p), read_text(p)) for p in args.resume]
    jobs = [(os.path.basename(p), read_text(p)) for p in args.job]

    results = []
    for rname, rtext in resumes:
        for jname, jtext in jobs:
            results.append((rname, jname, score_pair(emb, rtext, jtext)))

    results.sort(key=lambda x: -x[2]["total"])

    print("=" * 66)
    print("匹配结果（按总分从高到低）")
    print("=" * 66)
    for rname, jname, r in results:
        print(f"\n【{rname}  ×  {jname}】")
        print(f"  总分 {r['total']:.3f}  {bar(r['total'])}  → {r['verdict']}")
        print(f"  技能层 {r['skill']['score']:.2f}   语义层 {r['sem']:.3f}")
        ok = r["skill"]["hard_ok"]
        miss = r["skill"]["hard_miss"]
        print(f"  硬要求：命中 {('、'.join(ok) or '无')}")
        if miss:
            print(f"          缺失 {('、'.join(miss))}   ← 建议优先补这几项")
        soft = r["skill"]["soft_hit"]
        if soft:
            print(f"  加分项：命中 {('、'.join(soft))}")
        if args.detail or len(results) == 1:
            weak = [c for c in r.get("cov", []) if not c["covered"]]
            if weak:
                print("  简历没覆盖到的要求：")
                for c in weak:
                    print(f"    · {c['jd'][:34]}  (最接近的一句相似度 {c['sim']:.2f})")
                    print(f"      现有：{c['resume'][:34]}")
    print()


if __name__ == "__main__":
    main()

"""
评测诊断效果。两个数据集：
  data/eval_cases.json  57 条，开发时看过错例、据此补过关键词 → 偏乐观，当"开发集"
  data/seed_cases.json  38 条种子维修案例的描述 → 没拿来调过参数，当"留出集"，更接近真实水平

三种模式：只看文字；文字 + 模拟追问；（留出集额外）不含案例语料，避免"自己评自己"。

运行：python eval.py        加 -v 打印留出集错例
"""
import json
import os
import random
import sys

import db
import diagnoser
import engine
from diagnoser import HERE


def top1(ranker, cases):
    return sum(ranker.rank(c["text"])[0][0] == c["label"] for c in cases)


def top3(ranker, cases):
    return sum(c["label"] in [x for x, _ in ranker.rank(c["text"])[:3]] for c in cases)


def with_questions(eng, knowledge, cases, seed=1):
    """模拟技师：按该故障下的条件概率随机作答（不是挑对系统有利的答案）。"""
    rnd = random.Random(seed)
    ok = asked = 0
    for c in cases:
        ans = {}
        for _ in range(8):
            r = engine._Session(eng, knowledge).diagnose(text=c["text"], answers=ans)
            q = r["next_question"]
            if not q:
                break
            p = knowledge.q_by_id[q["id"]]["p"][c["label"]]
            keys = list(p)
            ans[q["id"]] = rnd.choices(keys, [p[k] for k in keys])[0]
        ok += r["candidates"][0]["id"] == c["label"]
        asked += len(ans)
    return ok, asked / len(cases)


def main(verbose=False):
    data = os.path.join(HERE, "data")
    faults = json.load(open(os.path.join(data, "faults.json"), encoding="utf-8"))
    names = {f["id"]: f["name"] for f in faults}
    dev = json.load(open(os.path.join(data, "eval_cases.json"), encoding="utf-8"))
    held = [{"text": c["text"], "label": c["fault_id"]} for c in json.load(open(os.path.join(data, "seed_cases.json"), encoding="utf-8"))]

    conn = db.init_db(os.path.join(data, "evdoctor.db"))
    eng = engine.Engine(conn)
    # 留出集必须用不含案例语料的模型，否则是拿案例评案例
    clean = engine._Knowledge(conn)
    clean.cases, clean.case_vecs = [], []
    clean.ranker = diagnoser.TfidfRanker(faults)

    print(f"模型：{clean.ranker.name}")
    print(f"{'':14s}{'样本':>6s}{'只看文字 Top-1':>16s}{'Top-3':>8s}{'文字+追问 Top-1':>17s}{'平均追问':>9s}")
    for label, cases in (("开发集", dev), ("留出集", held)):
        n = len(cases)
        t1, t3 = top1(clean.ranker, cases), top3(clean.ranker, cases)
        q1, nq = with_questions(eng, clean, cases)
        print(f"{label:14s}{n:>6d}{t1:>8d} {t1 / n:>6.1%}{t3 / n:>8.1%}{q1:>9d} {q1 / n:>6.1%}{nq:>9.1f}")

    if verbose:
        print("\n留出集错例（只看文字）：")
        for c in held:
            rk = clean.ranker.rank(c["text"])
            if rk[0][0] != c["label"]:
                pos = [x for x, _ in rk].index(c["label"]) + 1
                print(f"  [排第{pos}] {c['text']}\n           应为 {names[c['label']]}，给了 {names[rk[0][0]]}")


if __name__ == "__main__":
    main("-v" in sys.argv)

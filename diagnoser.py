"""
快速决策层（System One）：输入一句故障描述，秒出"各类故障的概率"。

现在用的是一个不用训练的基线：字符 n-gram TF-IDF + 故障码精确匹配 + 关键词命中，
三路打分加权后过 softmax 变成概率。纯 Python 标准库，没有任何依赖。

以后小 Jev 训好了，只要写一个类实现同样的 rank(text) -> [(fault_id, prob), ...]，
在 get_ranker() 里换掉就行，前后端一行都不用改。
"""
import json
import math
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FAULTS_PATH = os.path.join(HERE, "data", "faults.json")

# 故障码格式：P/B/C/U + 4 位十六进制，比如 P0A0A、U0111
# 注意不能用 \b：中文字符也算"单词字符"，"出来P0A0A" 中间没有词边界
DTC_RE = re.compile(r"(?<![A-Za-z0-9])([PBCU][0-9A-Fa-f]{4})(?![A-Za-z0-9])")


def load_faults(path=FAULTS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    """统一大小写、去掉空白和标点，方便做字符 n-gram。"""
    text = text.upper()
    text = re.sub(r"[\s，。、！？；：,.!?;:（）()\[\]【】“”\"'‘’~～\-—_/]+", "", text)
    return text


def char_ngrams(text, ns=(1, 2, 3)):
    """把一句话切成字符片段。中文不用分词，'上不了高压' -> '上','上不','上不了',..."""
    t = normalize(text)
    grams = []
    for n in ns:
        grams += [t[i:i + n] for i in range(len(t) - n + 1)]
    return grams


class BaseRanker:
    """所有快速决策模型的统一接口。"""
    name = "base"

    def rank(self, text):
        """返回按概率从大到小排好的 [(fault_id, prob), ...]，prob 总和为 1。"""
        raise NotImplementedError


class TfidfRanker(BaseRanker):
    name = "TF-IDF 基线"

    def __init__(self, faults, temperature=0.05):
        self.faults = faults
        self.ids = [f["id"] for f in faults]
        self.temperature = temperature  # 越小概率越"尖"，越大越"平"

        # 每类故障的"语料"：名称、关键词、症状、示例都算进去
        self.docs = {}
        for f in faults:
            parts = [f["name"], f["symptoms"]] + f["keywords"] + f["examples"]
            self.docs[f["id"]] = parts

        # 统计 IDF：一个片段在越少的故障类里出现，就越有区分度
        df = Counter()
        for fid, parts in self.docs.items():
            df.update(set(g for p in parts for g in char_ngrams(p)))
        n = len(faults)
        self.idf = {g: math.log((n + 1) / (c + 1)) + 1 for g, c in df.items()}

        # 每类故障一个"整体向量"，再给每条示例单独一个向量
        self.profile_vec = {}
        self.example_vecs = {}
        for f in faults:
            fid = f["id"]
            self.profile_vec[fid] = self._vec(" ".join(self.docs[fid]))
            self.example_vecs[fid] = [self._vec(e) for e in f["examples"] + f["keywords"]]

        # 关键词是否"专属"：没有出现在别的故障的关键词里（子串也算出现）
        self.kw_unique = {}
        for f in faults:
            others = [normalize(k) for g in faults if g["id"] != f["id"] for k in g["keywords"]]
            for k in f["keywords"]:
                nk = normalize(k)
                self.kw_unique[k] = not any(nk in o or o in nk for o in others if o)

        self.dtc_map = {}
        for f in faults:
            for code in f["dtc"]:
                self.dtc_map.setdefault(code.upper(), []).append(f["id"])

    def _vec(self, text):
        tf = Counter(char_ngrams(text))
        v = {g: (1 + math.log(c)) * self.idf.get(g, 0.0) for g, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    @staticmethod
    def _cos(a, b):
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(g, 0.0) for g, x in a.items())

    def scores(self, text):
        """返回每类故障的原始分数，以及分数里各部分的拆解（给前端解释用）。"""
        q = self._vec(text)
        codes = {c.upper() for c in DTC_RE.findall(text)}
        qn = normalize(text)
        out = {}
        for f in self.faults:
            fid = f["id"]
            s_profile = self._cos(q, self.profile_vec[fid])
            s_example = max((self._cos(q, ev) for ev in self.example_vecs[fid]), default=0.0)
            hits = [k for k in f["keywords"] if normalize(k) and normalize(k) in qn]
            # 专属关键词（如"电机过热"）加分多，几类故障共用的（如"冷却液""限功率"）加分少
            s_kw = min(sum(0.10 if self.kw_unique[k] else 0.04 for k in hits), 0.3)
            dtc_hit = [c for c in codes if fid in self.dtc_map.get(c, [])]
            s_dtc = 0.35 if dtc_hit else 0.0
            total = 0.45 * s_profile + 0.55 * s_example + s_kw + s_dtc
            out[fid] = {"score": total, "keywords": hits, "dtc": dtc_hit}
        return out

    def rank(self, text):
        raw = self.scores(text)
        mx = max(v["score"] for v in raw.values())
        exps = {fid: math.exp((v["score"] - mx) / self.temperature) for fid, v in raw.items()}
        z = sum(exps.values())
        # 平滑：留 3% 的概率均分给所有类。基线模型不该说"100% 确定"，
        # 文字描述本来就有歧义，给出 97% 比 100% 更诚实（思路同 label smoothing）
        eps, n = 0.03, len(exps)
        ranked = sorted(((fid, (1 - eps) * e / z + eps / n) for fid, e in exps.items()),
                        key=lambda x: -x[1])
        return ranked

    def explain(self, text):
        """rank 的加强版：顺带返回命中了哪些关键词 / 故障码，前端拿来高亮。"""
        raw = self.scores(text)
        return [
            {"id": fid, "prob": p, "keywords": raw[fid]["keywords"], "dtc": raw[fid]["dtc"]}
            for fid, p in self.rank(text)
        ]


class JevRanker(BaseRanker):
    """
    占位：小 Jev 训好以后在这里接入。
    思路：把故障描述喂给模型，让它在 19 个故障 id 上输出概率分布（不生成文字）。
    """
    name = "小 Jev"

    def __init__(self, faults, model_path):
        raise NotImplementedError("小 Jev 还没训好，先用 TF-IDF 基线。")


def get_ranker(faults=None):
    faults = faults or load_faults()
    jev_path = os.environ.get("EVDOC_JEV_MODEL")
    if jev_path:
        try:
            return JevRanker(faults, jev_path)
        except NotImplementedError as e:
            print("[提示]", e)
    return TfidfRanker(faults)


if __name__ == "__main__":
    r = get_ranker()
    faults = {f["id"]: f for f in load_faults()}
    while True:
        try:
            q = input("\n描述故障（回车退出）> ").strip()
        except EOFError:
            break
        if not q:
            break
        for item in r.explain(q)[:3]:
            f = faults[item["id"]]
            print(f"  {item['prob']*100:5.1f}%  {f['name']}  {item['dtc'] or ''} {item['keywords'] or ''}")

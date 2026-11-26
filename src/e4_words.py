"""Test 4: what does DINOv2's good/bad direction mean, in words?

1. Translate: fit a linear map from DINOv2-g (layer 37) to CLIP image embeddings on 5,000 unrated COCO photos
   (no SMID image, no rating). Apply it unchanged to SMID.
2. Check the translation: retrieval of each SMID image's true CLIP embedding, and a valence probe on translated features.
3. Read: for each word in a fixed vocabulary, correlate (across SMID images) the similarity between the translated image
   and "a photo of <word>" with DINOv2's own held-out probe score. Rank words; compare word groups with an AUC.

  python src/e4_words.py --cache CACHE --out results
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import Cache, cli, fit_map, oof, parse, rho, save_json, targets, unit

VOCAB = {
    "virtue": ["kindness", "compassion", "care", "generosity", "charity", "honesty", "fairness", "justice", "respect",
               "loyalty", "courage", "mercy", "forgiveness", "helping others", "a rescue", "volunteering", "sharing",
               "protection", "gratitude", "tenderness", "comfort", "self-sacrifice", "heroism", "nurturing",
               "cooperation", "solidarity", "dignity", "integrity", "decency", "humility"],
    "vice": ["cruelty", "violence", "abuse", "murder", "theft", "cheating", "lying", "betrayal", "corruption", "greed",
             "hatred", "torture", "bullying", "harassment", "exploitation", "vandalism", "terrorism", "assault",
             "robbery", "fraud", "neglect", "humiliation", "discrimination", "racism", "slavery", "oppression",
             "revenge", "injustice", "selfishness", "arrogance"],
    "positive emotion": ["joy", "happiness", "fun", "excitement", "calm", "pleasure", "delight", "relaxation",
                         "cheerfulness", "amusement", "laughter", "a party", "a vacation", "playfulness", "beauty",
                         "a celebration", "contentment", "a good meal", "a sunny day", "a game"],
    "negative emotion": ["sadness", "fear", "pain", "disgust", "boredom", "anxiety", "grief", "loneliness", "anger",
                         "misery", "a disaster", "illness", "dirt", "decay", "a storm", "an accident", "garbage",
                         "a funeral", "poverty", "exhaustion"],
    "thing": ["a dog", "a cat", "a horse", "a bird", "a car", "a bicycle", "a phone", "a computer", "a chair", "a table",
              "a cake", "a pizza", "a flower", "a tree", "a mountain", "a beach", "a city street", "a kitchen",
              "an office", "a train", "an airplane", "a boat", "a bridge", "a building", "a book", "a clock", "a lamp",
              "a shoe", "a bottle", "a market", "a road", "a river", "a forest", "a field", "a stadium", "a classroom",
              "a hospital", "a church", "a uniform", "a flag", "a crowd", "a child", "an old man", "a woman", "a soldier",
              "a doctor", "a nurse", "a police officer", "a weapon", "money"]}

GROUP_TESTS = [("virtue", "vice"), ("virtue", "positive emotion"), ("negative emotion", "vice")]


def text_embeddings(texts):
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    with torch.no_grad():
        out = model.get_text_features(**tok(texts, padding=True, return_tensors="pt"))
    if not torch.is_tensor(out):
        emb = getattr(out, "text_embeds", None)
        out = emb if emb is not None else out.pooler_output
    return unit(out.float().numpy().astype(np.float64))


def retrieval(translated, true, seed=0):
    """How close each translated SMID image lands to its own CLIP embedding, among all SMID images."""
    S = translated @ true.T
    n = len(S)
    rank = (S > S[np.arange(n), np.arange(n)][:, None]).sum(1)
    shuffled = np.random.default_rng(seed).permutation(n)
    return {"mean_cosine": round(float((translated * true).sum(1).mean()), 4),
            "mean_cosine_random_pairs": round(float((translated * true[shuffled]).sum(1).mean()), 4),
            "top1": round(float((rank == 0).mean()), 4), "top5": round(float((rank < 5).mean()), 4)}


def main():
    a = parse(cli(__doc__))
    cache = Cache(a.cache)
    T = targets(cache.labels())
    dino, clip = cache.dinov2(), cache.clip()
    Dc, Cc = cache.coco_pair()

    M = unit(fit_map(Dc, Cc, k=256).predict(dino))
    res = {"map": "linear, DINOv2-g layer 37 -> CLIP ViT-L/14 image embeddings, fit on 5,000 unrated COCO val2017 photos",
           "check": {"retrieval_512_component_map": retrieval(unit(fit_map(Dc, Cc, k=512).predict(dino)), clip),
                     "valence_probe_on_translated": rho(T["valence"], oof(M, T["valence"]))}}

    words = [(w, g) for g, ws in VOCAB.items() for w in ws]
    groups = np.array([g for _, g in words])
    sims = M @ text_embeddings([f"a photo of {w}" for w, _ in words]).T          # images x words
    for t, y in T.items():
        score = oof(dino, y)                                                       # DINOv2's own held-out score
        s = np.array([spearmanr(sims[:, j], score)[0] for j in range(len(words))])
        order = np.argsort(-s)
        auc = {f"{g1} > {g2}": round(float(roc_auc_score(np.r_[np.ones((groups == g1).sum()), np.zeros((groups == g2).sum())],
                                                         np.r_[s[groups == g1], s[groups == g2]])), 4) for g1, g2 in GROUP_TESTS}
        res[t] = {"top": [[words[j][0], words[j][1], round(float(s[j]), 4)] for j in order[:15]],
                  "bottom": [[words[j][0], words[j][1], round(float(s[j]), 4)] for j in order[::-1][:15]],
                  "auc": auc, "mean_by_group": {g: round(float(s[groups == g].mean()), 4) for g in VOCAB}}
        print(t, [w for w, _, _ in res[t]["top"][:8]], "|", [w for w, _, _ in res[t]["bottom"][:8]], auc, flush=True)
    save_json(res, f"{a.out}/e4_words.json")


if __name__ == "__main__":
    main()

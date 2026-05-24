# =========================
# STRICT Phrase Attention + ROBUST CSV LOADER + METRICS (ONE FILE)
# - Reads ALL lines robustly (repairs malformed CSV rows)
# - Phrase tokens glued with "_" (n=2..3)
# - TopK=3 chosen with early/mid/late spread + non-overlap
# - Outputs: top phrases + per-sample metrics + summary metrics
# - OUT_TOP now includes clean_tweet column right after Sample
# - Output files remain .csv BUT we use TAB as separator to avoid comma issues in text
#
# UPDATE (Option #1 Applied):
# - STRICT_BUCKETS now uses a QUALITY GATE per bucket:
#   Only pick from a bucket if candidate score >= BUCKET_MIN_REL * best_score
#   This avoids forcing weak early/mid phrases when evidence is concentrated (e.g., late IOC/CVE).
# =========================

import os
import re
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter

from tensorflow.keras.layers import (
    Layer, Input, Embedding, Bidirectional, LSTM, Dropout,
    Dense, LayerNormalization, MultiHeadAttention,
    GlobalMaxPooling1D, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer


# -------------------------
# 0) Settings
# -------------------------
CSV_PATH  = "twitter.csv"      # TSV file (despite name)
TEXT_COL  = "clean_tweet"
LABEL_COL = "relevant"

VOCAB_SIZE = 60000
MAX_LEN = 70
EMB_DIM = 128

TOPK_PHRASES = 3
NGRAM_MIN = 2
NGRAM_MAX = 3

USE_VENDOR_PENALTY = True
VENDOR_PENALTY_LAMBDA = 0.55

USE_TRIGGER_BONUS = True
TRIGGER_BONUS = 1.35

STRICT_BUCKETS = True
NON_OVERLAP = True
ALLOW_FALLBACK = True

# === Option 1: Bucket Quality Gate ===
# Only pick a phrase from a bucket if its score is at least this fraction of the best score in the sample.
# Typical range: 0.35 ~ 0.60
BUCKET_MIN_REL = 0.45

# output paths (keep .csv filenames)
OUT_TOP = "attention_top_phrases_twitter.csv"
OUT_PER = "per_sample_metrics_twitter.csv"
OUT_SUM = "metrics_summary_twitter.csv"

# IMPORTANT: keep filenames as .csv, but use TAB separator to avoid comma issues in tweet text
OUT_DELIM = "\t"


# -------------------------
# 1) Stopwords / Noise / Triggers / Vendors
# -------------------------
ALWAYS_REMOVE = {"<url>", "<email>", "<user>", "<num>", "<pct>", "<oov>"}

REMOVE_TOKENS = {
    "the","a","an","in","on","at","with","of","to","from","and","or","but","so","because",
    "into","over","under","about","before","after","during","around","within","across",
    "is","am","are","was","were","be","been","being",
    "have","has","had","do","does","did","can","could","will","would","should","may","might","must",
    "i","we","you","he","she","it","they","me","us","him","her","them",
    "my","your","his","its","our","their","this","that","these","those",
    "observed","observe","recent","recently","example","following","below","given",
    "round-up","roundup","demonstrate","demonstrates","trying","shows","show",
    "status","essential","popular","level","single","last","year","months","month",
    "q1","q2","q3","q4",
}

SECURITY_TRIGGERS = {
    "threat","threats","actor","actors","abuse","abusing",
    "phishing","ransomware","malware","botnet","ddos",
    "compromise","credential","credentials","oauth",
    "breach","intrusion","attack","attacks",
    "host","send","campaign","payload","payloads",
    "stealer","miner","keylogging","c2","cnc",
    "supply","chain",
}

VENDOR_WORDS = {
    "microsoft","google","azure","onedrive","sharepoint","office","365",
    "gsuite","g-suite","firebase","sendgrid",
    "outlook","outlookcom","outlook.com","sharepoint.com",
}


# -------------------------
# 2) Preprocess
# -------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", flags=re.IGNORECASE)
USER_RE = re.compile(r"@\w+", flags=re.IGNORECASE)
HASHTAG_RE = re.compile(r"#(\w+)")
PCT_RE = re.compile(r"\b\d+(\.\d+)?\s*%+\b")
NUM_RE = re.compile(r"\b\d+([\.,]\d+)*\b")

def preprocess_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [x](url) -> x
    text = text.replace("—","-").replace("–","-").replace("−","-").replace("’","'")
    text = re.sub(r"(\w)\'s\b", r"\1", text)

    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = USER_RE.sub(" <user> ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = PCT_RE.sub(" <pct> ", text)
    text = NUM_RE.sub(" <num> ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_word(tok: str) -> str:
    t = tok.strip().lower()
    t = t.replace("’","").replace("'","")
    t = re.sub(r"^[^\w<]+", "", t)
    t = re.sub(r"[^\w>]+$", "", t)
    t = re.sub(r"_+", "_", t)
    return t

def normalize_for_match(w: str) -> str:
    return w.replace(".","").replace("-","")

def is_noise_word(w: str) -> bool:
    if not w:
        return True
    if w in ALWAYS_REMOVE:
        return True
    if w in REMOVE_TOKENS:
        return True
    if re.fullmatch(r"[\W_]+", w):
        return True
    if re.fullmatch(r"\d+([.,]\d+)?", w):
        return True
    return False

def word_tokenize_keep_pos(text: str):
    words = []
    raw = text.split(" ")
    for idx, r in enumerate(raw):
        w = clean_word(r)
        if not w:
            continue
        words.append((w, idx))
    return words


# -------------------------
# 3) Robust TSV loader (reads ALL lines, repairs column count)
# -------------------------
def robust_read_tsv(path: str, text_col: str, label_col: str, delimiter="\t", encoding="utf-8"):
    """
    Reads TSV with csv.reader and repairs malformed rows:
      - if too many columns: joins extras into the text column (assumes label is last or known)
      - if too few columns: pads with empty strings
    Returns DataFrame + stats.
    """
    repairs = 0
    skipped_empty = 0
    total_data_lines = 0

    with open(path, "r", encoding=encoding, errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar='"', escapechar='\\')
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("File is empty.")

        # Normalize header names
        header = [h.strip() for h in header]
        if text_col not in header or label_col not in header:
            raise ValueError(f"Missing columns. Found header: {header}")

        idx_text = header.index(text_col)
        idx_label = header.index(label_col)

        rows = []
        for row in reader:
            # count raw data lines (non-empty)
            if len(row) == 0 or all(str(x).strip() == "" for x in row):
                skipped_empty += 1
                continue

            total_data_lines += 1

            # repair column count
            if len(row) != len(header):
                repairs += 1
                if len(row) > len(header):
                    # Join extra columns into the TEXT column (most common failure: tabs inside tweet)
                    while len(row) <= max(idx_text, idx_label):
                        row.append("")
                    fixed = [""] * len(header)
                    for i in range(len(header)):
                        if i < len(row):
                            fixed[i] = row[i]
                    overflow = row[len(header):]
                    if overflow:
                        fixed[idx_text] = (fixed[idx_text] + " " + " ".join(overflow)).strip()
                    row = fixed
                else:
                    # Too few columns: pad
                    row = row + [""] * (len(header) - len(row))

            d = {header[i]: row[i] if i < len(row) else "" for i in range(len(header))}
            rows.append(d)

    df = pd.DataFrame(rows)

    # cleanup types
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[df[label_col] != ""]
    df[label_col] = df[label_col].astype(int)
    df = df[df[text_col].str.strip() != ""]

    stats = {
        "total_data_lines": total_data_lines,
        "skipped_empty_lines": skipped_empty,
        "repairs": repairs,
        "final_rows": len(df)
    }
    return df, stats


# -------------------------
# 4) Build phrase tokens + spans
# -------------------------
def make_phrase_tokens_with_spans(words_with_pos, nmin=2, nmax=3):
    filtered = [(w, p) for (w, p) in words_with_pos if not is_noise_word(w)]
    toks = []
    spans = []
    L = len(filtered)
    for n in range(nmin, nmax + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            chunk = filtered[i:i+n]
            ws = [c[0] for c in chunk]
            ps = [c[1] for c in chunk]
            if any(w in ALWAYS_REMOVE for w in ws):
                continue
            phrase = "_".join(ws)
            toks.append(phrase)
            spans.append((min(ps), max(ps)))
    return toks, spans

def phrase_has_trigger(phrase_tok: str) -> bool:
    parts = [normalize_for_match(p) for p in phrase_tok.split("_")]
    return any(p in SECURITY_TRIGGERS for p in parts)

def vendor_ratio_phrase(phrase_tok: str) -> float:
    parts = [normalize_for_match(p) for p in phrase_tok.split("_")]
    if not parts:
        return 0.0
    hits = sum(1 for p in parts if p in VENDOR_WORDS)
    return hits / len(parts)

def pretty_phrase(phrase_tok: str) -> str:
    return phrase_tok.replace("_", " ")

def phrase_words_set(phrase_tok: str):
    return set(pretty_phrase(phrase_tok).split())


# -------------------------
# 5) Attention layer
# -------------------------
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight("att_weight", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight("att_bias", shape=(1,),
                                 initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x, mask=None):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        if mask is not None:
            mask = tf.cast(tf.expand_dims(mask, -1), tf.float32)
            a = a * mask
            a = a / (tf.reduce_sum(a, axis=1, keepdims=True) + 1e-8)
        out = tf.reduce_sum(a * x, axis=1)
        return out, a


# -------------------------
# 6) Load ALL lines robustly
# -------------------------
df, st = robust_read_tsv(CSV_PATH, TEXT_COL, LABEL_COL, delimiter="\t")

print("[INFO] robust_read_tsv stats:", st)
if st["final_rows"] != st["total_data_lines"]:
    print("[WARN] Some lines lacked label/text after repair & cleanup (e.g., missing label).")

# raw and preprocessed texts
raw_texts = df[TEXT_COL].astype(str).tolist()
texts = [preprocess_text(t) for t in raw_texts]
y = df[LABEL_COL].values


# -------------------------
# 7) Build phrase docs + spans
# -------------------------
phrase_docs = []
phrase_spans = []
doc_word_lengths = []

for t in texts:
    wpos = word_tokenize_keep_pos(t)
    toks, spans = make_phrase_tokens_with_spans(wpos, NGRAM_MIN, NGRAM_MAX)
    phrase_docs.append(toks)
    phrase_spans.append(spans)
    doc_word_lengths.append(max([p for _, p in wpos], default=0) + 1)

print("DEBUG first sample phrase tokens:", phrase_docs[0][:15])


# -------------------------
# 8) Tokenizer + aligned sequences (manual)
# -------------------------
tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters="", oov_token="<oov>")
tokenizer.fit_on_texts([" ".join(p) for p in phrase_docs])
oov_id = tokenizer.word_index.get("<oov>", 1)

def to_ids_and_meta(tokens, spans):
    ids = []
    meta = []
    for tok, sp in zip(tokens, spans):
        tid = tokenizer.word_index.get(tok, oov_id)
        if tid >= VOCAB_SIZE:
            tid = oov_id
        ids.append(tid)
        meta.append(sp)
    ids = ids[:MAX_LEN]
    meta = meta[:MAX_LEN]
    if len(ids) < MAX_LEN:
        pad_n = MAX_LEN - len(ids)
        ids.extend([0]*pad_n)
        meta.extend([None]*pad_n)
    return np.array(ids, dtype=np.int32), meta

X_list, META_list = [], []
for toks, spans in zip(phrase_docs, phrase_spans):
    ids, meta = to_ids_and_meta(toks, spans)
    X_list.append(ids)
    META_list.append(meta)

X = np.stack(X_list, axis=0)


# -------------------------
# 9) Model
# -------------------------
inp = Input(shape=(MAX_LEN,))
emb = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_DIM, mask_zero=True)(inp)

x = Bidirectional(LSTM(64, return_sequences=True))(emb)
x = Dropout(0.3)(x)

mha = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)
mha_out = mha(x, x)
x = LayerNormalization(epsilon=1e-6)(x + mha_out)

att_vec, att_weights = AttentionLayer()(x)

gmp = GlobalMaxPooling1D()(x)
gap = GlobalAveragePooling1D()(x)
feat = Concatenate()([gmp, gap, att_vec])

dense = Dense(128, activation="relu", kernel_regularizer=l2(1e-3))(feat)
dense = Dropout(0.3)(dense)
out = Dense(1, activation="sigmoid", kernel_regularizer=l2(1e-3))(dense)

model = Model(inp, out)
model.compile(optimizer=RMSprop(learning_rate=1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()


# -------------------------
# 10) Train
# -------------------------
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)

model.fit(
    X, y,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# -------------------------
# 11) IDF
# -------------------------
def compute_idf(docs):
    N = len(docs)
    dfc = {}
    for doc in docs:
        for t in set(doc):
            dfc[t] = dfc.get(t, 0) + 1
    idf = {}
    for t, d in dfc.items():
        idf[t] = np.log((N + 1) / (d + 1)) + 1.0
    return idf

IDF = compute_idf(phrase_docs)


# -------------------------
# 12) STRICT selection + metrics helpers
# -------------------------
def bucket_of_span(span, doc_len):
    if span is None or doc_len <= 0:
        return "mid"
    s, e = span
    center = (s + e) / 2.0
    r = center / max(doc_len - 1, 1)
    if r < 1/3:
        return "early"
    elif r < 2/3:
        return "mid"
    return "late"

def spans_overlap(a, b):
    if a is None or b is None:
        return False
    a0, a1 = a
    b0, b1 = b
    return not (a1 < b0 or b1 < a0)

def score_candidate(phrase_tok, att_val):
    idf = float(IDF.get(phrase_tok, 1.0))
    score = float(att_val) * idf

    if USE_TRIGGER_BONUS and phrase_has_trigger(phrase_tok):
        score *= TRIGGER_BONUS

    if USE_VENDOR_PENALTY:
        vr = vendor_ratio_phrase(phrase_tok)
        score *= (1.0 - VENDOR_PENALTY_LAMBDA * vr)

    return score

def try_pick_from_bucket(by_bucket, bucket_name, selected_items, selected_spans, selected_phrase_set):
    for c in by_bucket[bucket_name]:
        if c["phrase"] in selected_phrase_set:
            continue
        if NON_OVERLAP and any(spans_overlap(c["span"], sp) for sp in selected_spans):
            continue
        selected_items.append(c)
        selected_spans.append(c["span"])
        selected_phrase_set.add(c["phrase"])
        return True
    return False

def overlap_word_count(phrases):
    sets = [phrase_words_set(p) for p in phrases]
    if len(sets) < 2:
        return 0
    inter = set()
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            inter |= (sets[i] & sets[j])
    return len(inter)

def pos_spread_norm(spans, doc_len):
    centers = []
    for sp in spans:
        if sp is None:
            continue
        s, e = sp
        centers.append((s + e) / 2.0)
    if len(centers) <= 1:
        return 0.0
    spread = max(centers) - min(centers)
    denom = max(doc_len - 1, 1)
    return float(spread / denom)


# -------------------------
# 13) Extract attention + TopK + write outputs + metrics
# -------------------------
att_model = Model(inputs=model.input, outputs=att_weights)
A = att_model.predict(X, batch_size=128, verbose=1)

rows_top = []
rows_per = []
phrase_counter = Counter()

empty_samples = 0

for i in range(len(X)):
    ids = X[i]
    att = A[i].squeeze()
    meta_spans = META_list[i]
    doc_len = doc_word_lengths[i]

    cand = []
    for j in range(MAX_LEN):
        tid = int(ids[j])
        if tid == 0:
            continue
        p = tokenizer.index_word.get(tid, "")
        if not p or p in ALWAYS_REMOVE or p == "<oov>":
            continue
        if "_" not in p:
            continue

        span = meta_spans[j]
        bkt = bucket_of_span(span, doc_len)
        sc = score_candidate(p, float(att[j]))
        cand.append({"phrase": p, "score": sc, "bucket": bkt, "span": span})

    cand.sort(key=lambda d: d["score"], reverse=True)

    # buckets
    by_bucket = {"early": [], "mid": [], "late": []}
    for c in cand:
        by_bucket[c["bucket"]].append(c)

    # === Option 1: Bucket Quality Gate ===
    # Filter out weak candidates per bucket relative to the best score in this sample
    if cand:
        best_score = float(cand[0]["score"])
        # If best_score is zero/negative (rare), avoid filtering too aggressively
        if best_score > 0:
            thr = BUCKET_MIN_REL * best_score
            for b in ["early", "mid", "late"]:
                by_bucket[b] = [c for c in by_bucket[b] if c["score"] >= thr]
    else:
        best_score = 0.0

    selected_items = []
    selected_spans = []
    selected_phrase_set = set()

    if STRICT_BUCKETS:
        for b in ["early", "mid", "late"]:
            try_pick_from_bucket(by_bucket, b, selected_items, selected_spans, selected_phrase_set)

    # fill up
    if len(selected_items) < TOPK_PHRASES and ALLOW_FALLBACK:
        for c in cand:
            if c["phrase"] in selected_phrase_set:
                continue
            if NON_OVERLAP and any(spans_overlap(c["span"], sp) for sp in selected_spans):
                continue
            selected_items.append(c)
            selected_spans.append(c["span"])
            selected_phrase_set.add(c["phrase"])
            if len(selected_items) >= TOPK_PHRASES:
                break

    selected = [it["phrase"] for it in selected_items]
    selected_pretty = [pretty_phrase(p) for p in selected]
    selected_buckets = [bucket_of_span(it["span"], doc_len) for it in selected_items]

    # empty?
    empty = 1 if len(selected) == 0 else 0
    empty_samples += empty

    # metrics per sample
    ov = overlap_word_count(selected)
    ps = pos_spread_norm([it["span"] for it in selected_items], doc_len)
    bdiv = len(set(selected_buckets)) if selected_buckets else 0

    # OUT_TOP row (Sample + clean_tweet + TopPhrases)
    rows_top.append({
        "Sample": i + 1,
        "clean_tweet": raw_texts[i],  # raw from file (not preprocessed)
        "TopPhrases": " | ".join(selected_pretty)
    })

    rows_per.append({
        "Sample": i + 1,
        "n_phrases": len(selected),
        "empty": empty,
        "overlap_word_count": ov,
        "pos_spread_norm": ps,
        "bucket_diversity": bdiv,
        "buckets": ",".join(sorted(set(selected_buckets))) if selected_buckets else ""
    })

    # global phrase counts
    for p in selected_pretty:
        if p.strip():
            phrase_counter[p] += 1

# summary
num_samples = len(X)
total_selected_phrases = sum(r["n_phrases"] for r in rows_per)
empty_rate = float(empty_samples / max(num_samples, 1))

avg_overlap = float(np.mean([r["overlap_word_count"] for r in rows_per])) if rows_per else 0.0
avg_spread = float(np.mean([r["pos_spread_norm"] for r in rows_per])) if rows_per else 0.0
avg_bdiv = float(np.mean([r["bucket_diversity"] for r in rows_per])) if rows_per else 0.0

top_phr = phrase_counter.most_common(30)

summary_rows = []
summary_rows.append(("num_samples", num_samples))
summary_rows.append(("empty_samples", empty_samples))
summary_rows.append(("empty_rate", empty_rate))
summary_rows.append(("total_selected_phrases", total_selected_phrases))
summary_rows.append(("avg_overlap_word_count", avg_overlap))
summary_rows.append(("avg_pos_spread_norm", avg_spread))
summary_rows.append(("avg_bucket_diversity", avg_bdiv))
summary_rows.append(("loader_total_data_lines", st["total_data_lines"]))
summary_rows.append(("loader_repairs", st["repairs"]))
summary_rows.append(("loader_final_rows", st["final_rows"]))

# Also record the gate value used
summary_rows.append(("bucket_min_rel", BUCKET_MIN_REL))

for k, (ph, c) in enumerate(top_phr, start=1):
    summary_rows.append((f"top_phrase_{k}", ph))
    summary_rows.append((f"top_phrase_{k}_count", c))


# -------------------------
# 14) Write outputs (TAB-separated but .csv filenames)
# -------------------------
def write_delim_noquotes(df_out: pd.DataFrame, path: str, delimiter: str):
    df_out.to_csv(
        path,
        index=False,
        encoding="utf-8",
        sep=delimiter,
        quoting=csv.QUOTE_NONE,
        escapechar="\\"
    )

df_top = pd.DataFrame(rows_top)[["Sample", "clean_tweet", "TopPhrases"]]
df_per = pd.DataFrame(rows_per)
df_sum = pd.DataFrame(summary_rows, columns=["metric", "value"])

write_delim_noquotes(df_top, OUT_TOP, OUT_DELIM)
write_delim_noquotes(df_per, OUT_PER, OUT_DELIM)
write_delim_noquotes(df_sum, OUT_SUM, OUT_DELIM)

print("\nSaved outputs:")
print(" -", OUT_TOP, f"(sep={repr(OUT_DELIM)})")
print(" -", OUT_PER, f"(sep={repr(OUT_DELIM)})")
print(" -", OUT_SUM, f"(sep={repr(OUT_DELIM)})")

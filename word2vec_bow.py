import numpy as np
import pandas as pd
import regex as re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score

def tokenize(text: str):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

df = pd.read_csv("lab8_docs.tsv", sep="\t")
df = df.dropna(subset=["doc_id", "text", "label"]).reset_index(drop=True)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= 20].reset_index(drop=True)

docs_tokens = [tokenize(t) for t in df["text"].tolist()]

#Train Word2Vec
W2V_DIM = 100
print(f"Training Word2Vec dim={W2V_DIM} ...")

w2v = Word2Vec(
    sentences=docs_tokens,
    vector_size=W2V_DIM,
    window=5,
    min_count=2,
    workers=4,
    sg=1,          
    negative=10,
    epochs=10
)

words = list(w2v.wv.index_to_key)
W = np.vstack([w2v.wv[w] for w in words]).astype(np.float32)
print("Vocab size used for binning:", len(words))

Ks = [50, 100, 200]
results = []

for K in Ks:
    print(f"\nBinning words into K={K} clusters")
    km_words = KMeans(n_clusters=K, random_state=42, n_init="auto")
    word_bin = km_words.fit_predict(W) 

    w2bin = {w: int(b) for w, b in zip(words, word_bin)}

    X = np.zeros((len(docs_tokens), K), dtype=np.float32)

    for i, toks in enumerate(docs_tokens):
        bins = [w2bin[t] for t in toks if t in w2bin]
        if not bins:
            continue
        counts = np.bincount(bins, minlength=K).astype(np.float32)
        X[i] = counts / counts.sum()  

    Xn = normalize(X, norm="l2")

  
    km_docs = KMeans(n_clusters=2, random_state=42, n_init="auto")
    doc_labels = km_docs.fit_predict(Xn)

    sil = silhouette_score(Xn, doc_labels, metric="cosine")
    print("Silhouette (cosine):", sil)

    np.save(f"w2v_bow_K{K}_doc_embeddings.npy", Xn)
    results.append({"method": "Word2Vec+BoW(binned)", "K_bins": K, "w2v_dim": W2V_DIM, "silhouette_cosine": float(sil)})

res_df = pd.DataFrame(results).sort_values(["K_bins"])
res_df.to_csv("word2vec_bow_results.csv", index=False)
print("\nSaved: word2vec_bow_results.csv")
print(res_df)
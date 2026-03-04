import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import numpy as np

# load data
df = pd.read_csv("lab8_docs.tsv", sep="\t")

texts = df["text"].astype(str).tolist()

documents = [
    TaggedDocument(words=text.lower().split(), tags=[str(i)])
    for i, text in enumerate(texts)
]

configs = [
    {"name": "d2v_50", "vector_size": 50, "window": 5, "epochs": 20},
    {"name": "d2v_100", "vector_size": 100, "window": 5, "epochs": 20},
    {"name": "d2v_200", "vector_size": 200, "window": 8, "epochs": 30},
]

for cfg in configs:
    print("\nTraining", cfg["name"])

    model = Doc2Vec(
        vector_size=cfg["vector_size"],
        window=cfg["window"],
        min_count=2,
        workers=4,
        epochs=cfg["epochs"],
    )

    model.build_vocab(documents)

    model.train(
        documents,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    # extract embeddings
    embeddings = np.array([model.dv[str(i)] for i in range(len(documents))])

    np.save(f"{cfg['name']}_embeddings.npy", embeddings)

    model.save(f"{cfg['name']}.model")

print("\nDone.")
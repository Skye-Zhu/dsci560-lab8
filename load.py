import pandas as pd

df = pd.read_csv("lab8_docs.tsv", sep="\t")

df = df.dropna(subset=["doc_id", "text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() >= 20]

print("Rows:", len(df))
print("Unique labels:", df["label"].nunique())

print("\nExample rows:")
print(df.head(3))

print("\nTop labels:")
print(df["label"].value_counts().head(10))
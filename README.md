# dsci560-lab8
Representing Document Concepts with Embeddings
Team Name: zll
Members:
	•Benson Luo
	•Yanbing Zhu
	•Jicheng Liu

1. Overview
This project explores different approaches for representing Reddit posts as document embeddings and evaluating their effectiveness for clustering.
Two embedding methods are implemented:
	•	Doc2Vec
	•	Word2Vec + Bag-of-Words binning
The clustering performance of each method is evaluated using the silhouette score with cosine distance.

2. Dataset

The dataset consists of Reddit posts collected in Lab 5 and stored in a MySQL database.
Database: reddit_cluster
Data extraction query:

```sql
SELECT
  post_id AS doc_id,
  REPLACE(REPLACE(clean_text, '\r', ' '), '\n', ' ') AS text,
  subreddit AS label
FROM posts
WHERE clean_text IS NOT NULL
AND CHAR_LENGTH(clean_text) >= 20;
```

The results are exported to: lab8_docs.tsv

3. Environment Setup

Create environment:
```bash
conda create -n lab8 python=3.11
conda activate lab8
```
Install dependencies:
```bash
pip install pandas numpy scikit-learn gensim regex
```

4.running the code
```bash
python load.py
python doc2vec_train.py
python cluster_eval.py
python word2vec_bow.py
```

5.Output Files

Generated outputs include:
```
d2v_50_embeddings.npy
d2v_100_embeddings.npy
d2v_200_embeddings.npy

w2v_bow_K50_doc_embeddings.npy
w2v_bow_K100_doc_embeddings.npy
w2v_bow_K200_doc_embeddings.npy
```








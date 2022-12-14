# sparse-search

## Install

```shell
pip install sparse-search
```

## Guide

### Sparse Vectors

Text is converted into sparse vectors in three stages:

1. Tokenization: convert strings to tokens using libraries like [spacy](https://spacy.io)
   or [nltk](https://www.nltk.org)
2. Vectorization: convert tokens token frequencies 
3. BM25 Score: score token frequencies

```shell
pip install spacy datasets
python -m spacy download en_core_web_sm
```

Load Example Data
```python
from datasets import load_dataset

questions = load_dataset("BeIR/hotpotqa", 'queries', split='queries[:100]')
corpus = load_dataset("BeIR/hotpotqa", 'corpus', split='corpus[:10000]')
```
Fit BM25 Model
```python
from sparse_search import BM25
import spacy 

def tokenizer(text):
    return [token.text for token in nlp(text)]

nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

bm25 = BM25(tokenizer=tokenizer)
bm25.fit(corpus['text'])
```

## Hybrid Lexical + Semantic Text Search

Hybrid search combines dense and sparse vector search techniques.

```shell
pip install sentence_transformers
```

```python
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sparse_search import utils

nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

questions = load_dataset("BeIR/hotpotqa", 'queries', split='queries[:100]')
corpus = load_dataset("BeIR/hotpotqa", 'corpus', split='corpus[:10000]')


def tokenizer(text):
    return [token.text for token in nlp(text)]


bm25 = BM25(tokenizer=tokenizer)
bm25.fit(corpus['text'])

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

question = questions['text'][0]
print("Question:", question)
dense = model.encode([question]).tolist()[0]
sparse = bm25.transform_query(question)

hdense, hsparse = utils.hybrid_score_norm(dense, sparse, alpha=0.5)

...
```




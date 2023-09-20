from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_corpus = [
    tokenizer.tokenize(doc.open().read())
    for doc in tqdm(sorted(
        Path('fragments').glob('*.txt')
    ), desc='Reading documents')
]
tokenized_query = tokenizer.tokenize('Where should the ramps be put for wheelchair?')
bm25 = BM25Okapi(tokenized_corpus)

top_N = 5
scores = bm25.get_scores(tokenized_query)
results = [
    f'Score: {scores[idx]:.4f}\n'
    + tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenized_corpus[idx]))
    for idx in np.argsort(scores)[: -top_N - 1: -1]
]
divider = '\n' + '=' * 40 + '\n'
print(f'\nTop {top_N} results:')
print(divider + divider.join(results))
print()

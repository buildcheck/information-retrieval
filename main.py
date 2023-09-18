from pathlib import Path
from string import punctuation

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

tokenized_corpus = [doc.open().read().lower().split(' ') for doc in tqdm(list(Path('fragments').glob('*.txt')), desc='Reading documents')]
tokenized_query = 'Where should the ramps be put for wheelchair?'.lower().strip(punctuation).split(' ')
bm25 = BM25Okapi(tokenized_corpus)

top_N = 5
scores = bm25.get_scores(tokenized_query)
results = [f'Score: {scores[idx]:.4f}\n' + ' '.join(tokenized_corpus[idx]) for idx in np.argsort(scores)[: -top_N - 1: -1]]
divider = '\n' + '=' * 40 + '\n'
print(f'\nTop {top_N} results:')
print(divider + divider.join(results))
print()

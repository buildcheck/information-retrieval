from functools import partial
from pathlib import Path
import pickle
import time
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from tqdm import tqdm

def timefunc(msg, func):
    print(msg, end='')
    s = time.perf_counter()
    res = func()
    print(f'elapsed time: {time.perf_counter() - s:.4f} seconds')
    return res

def get_tokenized_corpus(tokenizer):
    save_file = Path('tokenized_corpus.pickle')
    if save_file.exists():
        tokenized_corpus = timefunc(
            'Loading tokenized documents... ',
            partial(pickle.load, save_file.open('rb'))
        )
    else:
        tokenized_corpus = [
            tokenizer.tokenize(doc.open().read())
            for doc in tqdm(sorted(
                Path('fragments').glob('*.txt')
            ), desc='Tokenizing documents')
        ]
        pickle.dump(tokenized_corpus, save_file.open('wb'))
    return tokenized_corpus

def get_results(tokenizer, scores, tokenized_corpus, top_N):
    return [
        f'Score: {scores[i]:.4f}\n'
        + tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenized_corpus[i]))
        for i in np.argsort(scores)[: -top_N - 1: -1]
    ]

def main(query):
    tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    tokenized_corpus = get_tokenized_corpus(tokenizer)

    tokenized_query = tokenizer.tokenize(query)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = timefunc(
        'Scoring query against corpus... ',
        partial(bm25.get_scores, tokenized_query)
    )
    top_N = 50
    results = timefunc('Decoding results... ', partial(
        get_results, tokenizer, scores, tokenized_corpus, top_N
    ))
    divider = '\n' + '=' * 40 + '\n'
    print(f'\nBM25 Top {top_N} results:')
    print(divider + divider.join(results))
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python3 {__file__} 'Message Here'", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

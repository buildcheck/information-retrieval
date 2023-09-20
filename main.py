from pathlib import Path
import pickle
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from transformers import BertTokenizer
from tqdm import tqdm

def main(query):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    save_file = Path('tokenized_corpus.pickle')
    if save_file.exists():
        print('Loading tokenized documents...')
        tokenized_corpus = pickle.load(save_file.open('rb'))
    else:
        tokenized_corpus = [
            tokenizer.tokenize(doc.open().read())
            for doc in tqdm(sorted(
                Path('fragments').glob('*.txt')
            ), desc='Tokenizing documents')
        ]
        pickle.dump(tokenized_corpus, save_file.open('wb'))
    tokenized_query = tokenizer.tokenize(query)
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

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python3 {__file__} 'Message Here'", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])

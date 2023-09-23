from functools import partial
from pathlib import Path
import pickle
import os
import time
import subprocess
import sys

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from tqdm import tqdm

DATA_DIR = Path('fragments')

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
                DATA_DIR.glob('*.txt')
            ), desc='Tokenizing documents')
        ]
        pickle.dump(tokenized_corpus, save_file.open('wb'))
    return tokenized_corpus

def get_version_hashes():
    data_hash = timefunc('Hashing data directory... ', partial(
        subprocess.run,
        f'tar cf - {DATA_DIR.name} | git hash-object --stdin',
        stdout=subprocess.PIPE, shell=True, text=True
    )).stdout.strip()
    commit_hash = timefunc('Getting code hash... ', partial(
        subprocess.run,
        'git rev-parse HEAD',
        stdout=subprocess.PIPE, shell=True, text=True
    )).stdout.strip()
    return data_hash, commit_hash

def output_print(top_docs, top_scores):
    results = [
        f'Score: {top_scores[i]:.4f}\n' + top_docs[i]
        for i in range(top_scores.shape[0])
    ]
    divider = '\n' + '=' * 40 + '\n'
    print(f'\nBM25 Top {top_scores.shape[0]} results:')
    print(divider + divider.join(results))
    print()

def output_csv(top_docs, top_scores, qcsv):
    df_docs = pd.DataFrame(top_docs, columns=[
        f"Doc {i}" for i in range(1, top_scores.shape[1] + 1)
    ])
    df_scores = pd.DataFrame(top_scores, columns=[
        f"BM25 Score {i}" for i in range(1, top_scores.shape[1] + 1)
    ])

    # Interleave the two DataFrames and add blank columns
    columns = []
    for i in range(1, df_docs.shape[1] + 1):
        columns.append(df_docs[f"Doc {i}"])
        columns.append(pd.Series(
            np.nan, name=f"Expert Score {i}", index=df_docs.index
        ))
        columns.append(df_scores[f"BM25 Score {i}"])

    # Concatenate along the columns axis
    df_interleaved = pd.concat(columns, axis=1)
    print('Outputting to csv...')
    outfile = Path('ir-answers.csv').open('w')
    data_hash, commit_hash = get_version_hashes()
    outfile.write(f'Data version: {data_hash},Code version: {commit_hash}\n')
    pd.concat([qcsv, df_interleaved], axis=1).to_csv(outfile, index=False)
    print('Success!')

def main(query):
    '''
    If the user provided a query, then get the answers for it and print results
    Otherwise, read queries from csv and output to csv
    '''
    tokenizer = AutoTokenizer.from_pretrained('colbert-ir/colbertv2.0')
    tokenized_corpus = get_tokenized_corpus(tokenizer)
    bm25 = timefunc('Initializing BM25... ', partial(BM25Okapi, tokenized_corpus))

    if query:
        tokenized_queries = [tokenizer.tokenize(query)]
    else:
        # Since the user did not provide a query, we instead read queries from CSV
        qcsv = pd.read_csv('questions.csv')
        tokenized_queries = [tokenizer.tokenize(q) for q in qcsv['Question Content']]

    scores = np.array([timefunc(
        f'Scoring query {i} against corpus... ',
        partial(bm25.get_scores, tokenized_query)
    ) for i, tokenized_query in enumerate(tokenized_queries)])

    top_N = 50
    top_idx = np.argsort(scores)[:, : -top_N - 1: -1]

    doc_filenames = sorted(DATA_DIR.glob('*.txt'))
    # We could generate this by using `tokenizer.decode` on the tokenized
    # corpus, but it doesn't reconstruct originals perfectly (e.g.
    # capitalization is all lower case) so we read again directly the originals
    top_docs = [[
        f'Source file: {doc_filenames[i].name}\n'
        + (d := doc_filenames[i].open().read())[d.index('\n') + 1:] for i in r
    ] for r in top_idx]
    top_scores = scores[np.arange(scores.shape[0])[:, None], top_idx]

    if query:
        output_print(top_docs[0], top_scores[0])
    else:
        output_csv(top_docs, top_scores, qcsv)

if __name__ == '__main__':
    # Avoids potential deadlocks
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main(sys.argv[1] if len(sys.argv) >= 2 else None)

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
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

DATA_DIR = Path('fragments')
HF_MODEL_NAME = 'colbert-ir/colbertv2.0'

def timefunc(msg, func):
    print(msg, end='', flush=True)
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

def get_colbert_processed_corpus(model, tokenizer, max_sequence_length, tokenized_corpus):
    '''
    ColBERT represents every document as a matrix
    '''
    save_file = Path('colbert_processed_corpus.pt')
    if save_file.exists():
        matrices = timefunc(
            'Loading ColBERT processed corpus... ',
            partial(torch.load, save_file.open('rb'))
        )
    else:
        tokenized_ids = tokenizer.pad({"input_ids": [
            # Turn all word pieces into ids so the model can ingest them.
            tokenizer.convert_tokens_to_ids(doc_word_pieces)[:max_sequence_length]
            for doc_word_pieces in tokenized_corpus
        ]}, return_tensors='pt')
        with torch.inference_mode():
            batch_size = 100
            loader = DataLoader(
                TensorDataset(tokenized_ids['input_ids'][:300], tokenized_ids['attention_mask'][:300]),
                batch_size=batch_size,
                num_workers=1,
                shuffle=False
            )
            matrices = torch.cat([
                model(input_ids=inp, attention_mask=att)['last_hidden_state']
                for inp, att in tqdm(
                    loader, desc=(
                        'Preprocessing corpus with ColBERT with batch size of '
                        f'{batch_size} documents'
                    )
                )
            ])
        timefunc(
            'Saving ColBERT-processed corpus as matrices... ',
            partial(torch.save, matrices, save_file.open('wb'))
        )
    return matrices

# Ensure gradients can't be stored on the model
# (they are useless since we are not training)
@torch.inference_mode()
def get_scores(tokenized_queries, tokenized_corpus, tokenizer):

    if len(sys.argv) < 2 or sys.argv[1] == 'bm25':
        bm25 = timefunc('Initializing BM25... ', partial(BM25Okapi, tokenized_corpus))

        def scorer(tokenized_query):
            return bm25.get_scores(tokenized_query)

    elif sys.argv[1] == 'colbert':

        model = AutoModel.from_pretrained(HF_MODEL_NAME)
        model.pooler = None

        # Truncate max sequence length to 128 because less than 4% of
        # docs are more than this, and truncation reduces compute.
        max_sequence_length = 128
        matrices = get_colbert_processed_corpus(
            model, tokenizer, max_sequence_length, tokenized_corpus
        )

        def scorer(tokenized_query):
            input_ids = [tokenizer.convert_tokens_to_ids(tokenized_query)[:max_sequence_length]]

            # TODO Should I make tensor contiguous?
            # `.contiguous()` after transpose `.T` should presumably make
            # future computations a bit more efficient because `.T` by itself
            # causes the tensor to be laid out in memory in a way not very
            # efficient for normal computation. But it is not clear if matrix
            # multiply is "normal computation":
            # https://chat.openai.com/share/cc0af7dd-fe1f-40dd-a412-0933f5d1bfaf
            q = model(
                **tokenizer.pad({"input_ids": input_ids}, return_tensors='pt')
            ).last_hidden_state.reshape(len(input_ids[0]), -1).T
            return (matrices @ q).max(axis=1)[0].sum(axis=1).numpy()
    else:
        raise ValueError(
            f'Command line option "{sys.argv[1]}" is not a valid argument'
        )
    return np.stack([
        scorer(tokenized_query) for tokenized_query in tqdm(
            tokenized_queries, desc='Scoring query against corpus'
        )
    ])

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
    print(f'\nTop {top_scores.shape[0]} results:')
    print(divider + divider.join(results))
    print()

def output_csv(top_docs, top_scores, qcsv):
    df_docs = pd.DataFrame(top_docs, columns=[
        f"Doc {i}" for i in range(1, top_scores.shape[1] + 1)
    ])
    df_scores = pd.DataFrame(top_scores, columns=[
        f"Model Score {i}" for i in range(1, top_scores.shape[1] + 1)
    ])

    # Interleave the two DataFrames and add blank columns
    columns = []
    for i in range(1, df_docs.shape[1] + 1):
        columns.append(df_docs[f"Doc {i}"])
        columns.append(pd.Series(
            np.nan, name=f"Expert Score {i}", index=df_docs.index
        ))
        columns.append(df_scores[f"Model Score {i}"])

    # Concatenate along the columns axis
    df_interleaved = pd.concat(columns, axis=1)
    print('Outputting to csv...')
    outfile = Path(f'ir-answers-{sys.argv[1] if len(sys.argv) > 1 else "bm25"}.csv').open('w')
    data_hash, commit_hash = get_version_hashes()
    outfile.write(f'Data version: {data_hash},Code version: {commit_hash}\n')
    pd.concat([qcsv, df_interleaved], axis=1).to_csv(outfile, index=False)
    print('Success!')

def main(query):
    '''
    If the user provided a query, then get the answers for it and print results
    Otherwise, read queries from csv and output to csv
    '''
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    tokenized_corpus = get_tokenized_corpus(tokenizer)

    if query:
        tokenized_queries = [tokenizer.tokenize(query)]
    else:
        # Since the user did not provide a query, we instead read queries from CSV
        qcsv = pd.read_csv('questions.csv')
        tokenized_queries = [tokenizer.tokenize(q) for q in qcsv['Question Content']]

    scores = get_scores(tokenized_queries, tokenized_corpus, tokenizer)
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
    main(None)

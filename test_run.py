import os
from argparse import ArgumentParser
from itertools import product
from collections import OrderedDict

from ngram import NGram, NGramDataset
from pfsa import PFSA, PFSADataset
from pcfg import PCFG, PCFGDataset
from lm import train_model

# constant over all training runs
DEBUG = False
N_HEAD = 4 if not DEBUG else 2 # ignored by LSTM
MAX_LENGTH = 128
MAX_EPOCHS = 20 if not DEBUG else 1
LR = 1e-3
WD = 1e-5
LOG_FREQ = 100 if not DEBUG else 10
EVAL_EVERY = 1000 if not DEBUG else 100
NUM_SEQS_TRAIN = 10_000 if not DEBUG else 1000
NUM_SEQS_VAL = 10_000 if not DEBUG else 1000

# model-specific
N_EMBD_LSTM = 128 if not DEBUG else 64
N_EMBD_TRF = 256 if not DEBUG else 64

N_HIDDEN_LSTM = 256 if not DEBUG else 64
N_HIDDEN_TRF = 128 if not DEBUG else 64

N_LAYER_LSTM = 6 if not DEBUG else 3
N_LAYER_TRF = 4 if not DEBUG else 3

# constant over formalisms
seeds = [0]
nums_symbols = [10]
entropies = [4.]
model_types = ['lstm', 'trf']

# formalism-specific
ngram_orders = [2]
pfsa_nums_states = [8]
pcfg_nums_nts = [8]

default_grid = OrderedDict({
    'seed': seeds,
    'num_symbols': nums_symbols,
    'entropies': entropies
})

ngram_grid = default_grid.copy()
ngram_grid['order'] = ngram_orders

pfsa_grid = default_grid.copy()
pfsa_grid['num_states'] = pfsa_nums_states

pcfg_grid = default_grid.copy()
pcfg_grid['num_non_terminals'] = pcfg_nums_nts

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        '-j',
        type=int,
        required=True
    )
    args = parser.parse_args()
    j = args.j

    if j == 0:
        dataset_type = NGramDataset
        grid = ngram_grid
    elif j == 1:
        dataset_type = PFSADataset
        grid = pfsa_grid
    elif j == 2:
        dataset_type = PCFGDataset
        grid = pcfg_grid
    else:
        raise ValueError(f'j must be 0-2 but was {j}')

    for grammar_args in product(*grid.values()):

        seed, num_symbols, entropy, formalism_arg = grammar_args

        if j == 0:
            grammar = NGram(
                seed=seed,
                num_symbols=num_symbols,
                order=formalism_arg
            )
        elif j == 1:
            grammar = PFSA(
                seed=seed,
                num_symbols=num_symbols,
                num_states=formalism_arg
            )
        elif j == 2:
            grammar = PCFG(
                seed=seed,
                num_symbols=num_symbols,
                num_non_terminals=formalism_arg
            )

        print(f'Optimizing {grammar} to have entropy {entropy}...', flush=True)
        if not grammar.optimize(H_t=entropy, do_logging=True):
            print(f'Optimization failed. Consider retrying.', flush=True)
            if not os.path.exists('failed.tsv'):
                open('failed.tsv', 'w+', encoding='utf-8')
            with open('failed.tsv', 'r', encoding='utf-8') as f:
                line = f'{grammar}\t{entropy}'
                if line not in f.read().splitlines():
                    f.close()
                    with open('failed.tsv', 'a', encoding='utf-8') as g:
                        g.write(line + '\n')
            continue
        
        print(f'True entropy: {entropy}.', flush=True)
        ge = grammar.entropy().item()
        print(f'Grammar entropy: {ge}', flush=True)
        print(f'Diff: {abs(ge - entropy)}', flush=True)
        
        print(f'Generating {NUM_SEQS_TRAIN:,} sequences with {grammar}...', flush=True)
        train_data = dataset_type(grammar, num_seqs=NUM_SEQS_TRAIN, max_length=MAX_LENGTH)
        print(f'Generating {NUM_SEQS_VAL:,} sequences with {grammar}...', flush=True)
        val_data = dataset_type(grammar, num_seqs=NUM_SEQS_VAL, max_length=MAX_LENGTH)

        print(f'Training LSTM:', flush=True)
        print(f'\tn_embd: {N_EMBD_LSTM}', flush=True)
        print(f'\tn_hidden: {N_HIDDEN_LSTM}', flush=True)
        print(f'\tn_layer: {N_LAYER_LSTM}', flush=True)
        print(f'\tn_head: {N_HEAD}', flush=True)
        print(f'\tn_positions: {MAX_LENGTH}', flush=True)
        print(f'\tlr: {LR}', flush=True)
        print(f'\twd: {WD}', flush=True)
        print(f'\tmax_epochs: {MAX_EPOCHS}', flush=True)
        print(f'\tlog_freq: {LOG_FREQ}', flush=True)
        print(f'\teval_every: {EVAL_EVERY}', flush=True)
        train_model(
            grammar,
            train_data,
            val_data,
            n_embd = N_EMBD_LSTM,
            n_hidden = N_HIDDEN_LSTM,
            n_layer = N_LAYER_LSTM,
            n_head = N_HEAD,
            n_positions = MAX_LENGTH,
            lr = LR,
            wd = WD,
            max_epochs = MAX_EPOCHS,
            log_freq = LOG_FREQ,
            eval_every = EVAL_EVERY,
            trf_or_lstm = 'lstm',
            is_test_run=True
        )

        print(f'Training TRF:', flush=True)
        print(f'\tn_embd: {N_EMBD_TRF}', flush=True)
        print(f'\tn_hidden: {N_HIDDEN_TRF}', flush=True)
        print(f'\tn_layer: {N_LAYER_TRF}', flush=True)
        print(f'\tn_head: {N_HEAD}', flush=True)
        print(f'\tn_positions: {MAX_LENGTH}', flush=True)
        print(f'\tlr: {LR}', flush=True)
        print(f'\twd: {WD}', flush=True)
        print(f'\tmax_epochs: {MAX_EPOCHS}', flush=True)
        print(f'\tlog_freq: {LOG_FREQ}', flush=True)
        print(f'\teval_every: {EVAL_EVERY}', flush=True)
        train_model(
            grammar,
            train_data,
            val_data,
            n_embd = N_EMBD_TRF,
            n_hidden = N_HIDDEN_TRF,
            n_layer = N_LAYER_TRF,
            n_head = N_HEAD,
            n_positions = MAX_LENGTH,
            lr = LR,
            wd = WD,
            max_epochs = MAX_EPOCHS,
            log_freq = LOG_FREQ,
            eval_every = EVAL_EVERY,
            trf_or_lstm = 'trf',
            is_test_run=True
        )
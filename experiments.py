import os
import sys
from argparse import ArgumentParser
from itertools import product
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading
import logging

from ngram import NGram, NGramDataset
from pfsa import PFSA, PFSADataset
from pcfg import PCFG, PCFGDataset
from lm import train_model

logger = logging.getLogger(__name__)
_experiment_lock = threading.Lock()

# constant over all training runs
DEBUG = False
N_HEAD = 4 if not DEBUG else 2 # ignored by LSTM
MAX_LENGTH = 128
MAX_EPOCHS = 20 if not DEBUG else 1
LR = 1e-3
WD = 1e-5
LOG_FREQ = 100 if not DEBUG else 10
EVAL_EVERY = 100 if not DEBUG else 100
NUM_SEQS_TRAIN = 100_000 if not DEBUG else 1000
NUM_SEQS_VAL = 100_000 if not DEBUG else 1000

# model-specific
N_EMBD_LSTM = 128 if not DEBUG else 64
N_EMBD_TRF = 256 if not DEBUG else 64

N_HIDDEN_LSTM = 256 if not DEBUG else 64
N_HIDDEN_TRF = 128 if not DEBUG else 64

N_LAYER_LSTM = 6 if not DEBUG else 3
N_LAYER_TRF = 4 if not DEBUG else 3

# constant over formalisms
seeds = [0]#, 1, 2, 3, 4]
nums_symbols = [10, 100, 1000, 10_000, 100_000]
entropies = [4., 8., 16., 32., 64.]
model_types = ['lstm', 'trf']

# formalism-specific
ngram_orders = [1, 2, 3, 4, 5]
pfsa_nums_states = [4, 8, 16, 32, 64]
pcfg_nums_nts = [4, 8, 16, 32, 64]

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

def main(grammar_args, j):

    if not os.path.exists('experiments'):
        os.makedirs('experiments', exist_ok=True)
    
    with _experiment_lock:
        # All of this happens atomically - only one thread at a time
        if not os.path.exists('experiments'):
            this_experiment = '1'
        else:
            experiments = [int(x.replace('_test', '')) for x in os.listdir('experiments')]
            this_experiment = str(max(experiments) + 1)
        if DEBUG:
            this_experiment += '_test'
        this_experiment_dir = os.path.join('experiments', this_experiment)
        os.makedirs(this_experiment_dir)

    # TODO: set up parallel logging for different experiments

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

    logger.info(f'Optimizing {grammar} to have entropy {entropy}...')
    if not grammar.optimize(H_t=entropy, do_logging=True):
        logger.info(f'Optimization failed. Consider retrying.')
        with _experiment_lock:
            if not os.path.exists('failed.tsv'):
                open('failed.tsv', 'w+', encoding='utf-8')
            with open('failed.tsv', 'r', encoding='utf-8') as f:
                line = f'{grammar}\t{entropy}'
                if line not in f.read().splitlines():
                    f.close()
                    with open('failed.tsv', 'a', encoding='utf-8') as g:
                        g.write(line + '\n')
            return
    
    logger.info(f'True entropy: {entropy}.')
    ge = grammar.entropy().item()
    logger.info(f'Grammar entropy: {ge}')
    logger.info(f'Diff: {abs(ge - entropy)}')
    
    logger.info(f'Generating {NUM_SEQS_TRAIN:,} sequences with {grammar}...')
    train_data = dataset_type(grammar, num_seqs=NUM_SEQS_TRAIN, max_length=MAX_LENGTH)
    logger.info(f'Generating {NUM_SEQS_VAL:,} sequences with {grammar}...')
    val_data = dataset_type(grammar, num_seqs=NUM_SEQS_VAL, max_length=MAX_LENGTH)

    logger.info(f'Training LSTM:')
    logger.info(f'\tn_embd: {N_EMBD_LSTM}')
    logger.info(f'\tn_hidden: {N_HIDDEN_LSTM}')
    logger.info(f'\tn_layer: {N_LAYER_LSTM}')
    logger.info(f'\tn_head: {N_HEAD}')
    logger.info(f'\tn_positions: {MAX_LENGTH}')
    logger.info(f'\tlr: {LR}')
    logger.info(f'\twd: {WD}')
    logger.info(f'\tmax_epochs: {MAX_EPOCHS}')
    logger.info(f'\tlog_freq: {LOG_FREQ}')
    logger.info(f'\teval_every: {EVAL_EVERY}')
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
        this_experiment_dir = this_experiment_dir
    )

    logger.info(f'Training TRF:')
    logger.info(f'\tn_embd: {N_EMBD_TRF}')
    logger.info(f'\tn_hidden: {N_HIDDEN_TRF}')
    logger.info(f'\tn_layer: {N_LAYER_TRF}')
    logger.info(f'\tn_head: {N_HEAD}')
    logger.info(f'\tn_positions: {MAX_LENGTH}')
    logger.info(f'\tlr: {LR}')
    logger.info(f'\twd: {WD}')
    logger.info(f'\tmax_epochs: {MAX_EPOCHS}')
    logger.info(f'\tlog_freq: {LOG_FREQ}')
    logger.info(f'\teval_every: {EVAL_EVERY}')
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
        this_experiment_dir = this_experiment_dir
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-j',
        type=int,
        required=True,
        help='0 for n-grams, 1 for PFSAs, 2 for PCFGs.'
    )
    parser.add_argument(
        '--max_threads',
        type=int,
        default=8,
        help='Number of threads to run experiments in parallel.'
    )
    args = parser.parse_args()
    j = args.j
    max_threads = args.max_threads

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

    # can go more than 1 if we get that nice GPU ready
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(main, grammar_args, j)
            for grammar_args in product(*grid.values())
        ]
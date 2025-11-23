import os
import shutil
from argparse import ArgumentParser
from itertools import product
from collections import OrderedDict

import torch

from ngram import NGram, NGramDataset
from pfsa import PFSA, PFSADataset
from pcfg import PCFG, PCFGDataset
from lm import train_model

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

# constant over all training runs
DEBUG = True
PATIENCE = 5 # number of evals to wait before breaking if no appreciable change
TOLERANCE = 1e-3 # proportion of loss decrease equivalent to "no appreciable change"
VERBOSE = False
BATCH_SIZE = 32
N_HEAD = 4 if not DEBUG else 2 # ignored by LSTM
MAX_LENGTH = 128
MAX_EPOCHS = 20 if not DEBUG else 20
LR = 1e-3
WD = 1e-5
LOG_FREQ = 100 if not DEBUG else 1
EVAL_EVERY = 100 if not DEBUG else 1
NUM_SEQS_TRAIN = 128_000 if not DEBUG else 256
NUM_SEQS_VAL = 128_000 if not DEBUG else 256

# model-specific
N_EMBD_LSTM = 128 if not DEBUG else 64
N_EMBD_TRF = 256 if not DEBUG else 64

N_HIDDEN_LSTM = 256 if not DEBUG else 64
N_HIDDEN_TRF = 128 if not DEBUG else 64

N_LAYER_LSTM = 6 if not DEBUG else 3
N_LAYER_TRF = 4 if not DEBUG else 3

# formalism-specific
ngram_orders = [3, 4, 5]
pfsa_nums_states = [16, 32, 64]
pcfg_nums_nts = [16, 32, 64]

# constant over formalisms
default_grid = OrderedDict({
    'seed': [0],
    'num_symbols': [1000, 10_000, 100_000],
    'entropies': [16., 32., 64.]
})

ngram_grid = default_grid.copy()
ngram_grid['order'] = ngram_orders

pfsa_grid = default_grid.copy()
pfsa_grid['num_states'] = pfsa_nums_states

pcfg_grid = default_grid.copy()
pcfg_grid['num_non_terminals'] = pcfg_nums_nts

def main(grammar_args, j):
    
    if not os.path.exists('experiments'):
        this_experiment = '1'
    else:
        experiments = [int(x.replace('_test', '').replace('_lstm', '').replace('_trf', '')) for x in os.listdir('experiments')]
        this_experiment = str(max(experiments) + 1)
    if DEBUG:
        this_experiment += '_test'
    this_experiment_dir = os.path.join('experiments', this_experiment)
    os.makedirs(this_experiment_dir)

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

    grammar = grammar.to(DEVICE)
    print(f'Optimizing {grammar} on {DEVICE} to have entropy {entropy}...')
    if not grammar.optimize(H_t=entropy, do_logging=True):
        print(f'Optimization failed. Consider retrying.')
        if not os.path.exists('failed_opt.tsv'):
            open('failed_opt.tsv', 'w+', encoding='utf-8')
        with open('failed_opt.tsv', 'r', encoding='utf-8') as f:
            line = f'{grammar}\t{entropy}'
            if line not in f.read().splitlines():
                f.close()
                with open('failed_opt.tsv', 'a', encoding='utf-8') as g:
                    g.write(line + '\n')
    grammar = grammar.to('cpu')
    
    print(f'Target entropy: {entropy}.')
    ge = grammar.entropy().item()
    print(f'Grammar entropy: {ge}')
    print(f'Diff: {abs(ge - entropy)}')
    
    print(f'Generating {NUM_SEQS_TRAIN:,} sequences with {grammar}...')
    train_data = dataset_type(
        grammar,
        num_seqs=NUM_SEQS_TRAIN,
        max_length=MAX_LENGTH,
        do_logging=False,
        data_dir=os.path.join(this_experiment_dir, 'train')
    )
    print(f'Generating {NUM_SEQS_VAL:,} sequences with {grammar}...')
    val_data = dataset_type(
        grammar,
        num_seqs=NUM_SEQS_VAL,
        max_length=MAX_LENGTH,
        do_logging=False,
        data_dir=os.path.join(this_experiment_dir, 'val')
    )
    
    print(f'Computing train excess entropy...')
    train_ee = train_data.excess_entropy()
    print(f'Computing val excess entropy...')
    val_ee = val_data.excess_entropy()
    print(f'Computing basic stats...')
    hparams = {
        'grammar_type': grammar.formalism,
        'grammar_seed': grammar.seed,
        'grammar_num_symbols': grammar.num_symbols,
        'grammar_str': grammar.file_name_convention,
        'grammar_formalism_arg': formalism_arg,
        'grammar_target_entropy': entropy,
        'grammar_actual_entropy': grammar.entropy().item(),
        'train_data_stats': train_data.basic_stats(),
        'train_data_ee': train_ee,
        'val_data_stats': val_data.basic_stats(),
        'val_data_ee': val_ee, 
        'n_embd': N_EMBD_LSTM,
        'n_hidden': N_HIDDEN_LSTM,
        'n_layer': N_LAYER_LSTM,
        'n_head': N_HEAD,
        'n_positions': MAX_LENGTH,
        'lr': LR,
        'wd': WD,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'eval_every': EVAL_EVERY,
        'log_freq': LOG_FREQ,
        'model_type': 'lstm',
        'patience': PATIENCE,
        'tolerance': TOLERANCE
    }

    if not os.path.exists(os.path.join(this_experiment_dir, 'lstm')):
        print(f'Training LSTM:')
        print(f'\tn_embd: {N_EMBD_LSTM}')
        print(f'\tn_hidden: {N_HIDDEN_LSTM}')
        print(f'\tn_layer: {N_LAYER_LSTM}')
        print(f'\tn_head: {N_HEAD}')
        print(f'\tn_positions: {MAX_LENGTH}')
        print(f'\tlr: {LR}')
        print(f'\twd: {WD}')
        print(f'\tmax_epochs: {MAX_EPOCHS}')
        print(f'\tlog_freq: {LOG_FREQ}')
        print(f'\teval_every: {EVAL_EVERY}')
        train_model(
            grammar,
            train_data,
            val_data,
            hparams,
            this_experiment_dir = os.path.join(this_experiment_dir, 'lstm'),
            verbose = VERBOSE
        )
    
    if not os.path.exists(os.path.join(this_experiment_dir, 'trf')):
        hparams['model_type'] = 'trf'
        hparams['n_embd'] = N_EMBD_TRF
        hparams['n_hidden'] = N_HIDDEN_TRF
        hparams['n_layer'] = N_LAYER_TRF
        
        print(f'Training TRF:')
        print(f'\tn_embd: {N_EMBD_TRF}')
        print(f'\tn_hidden: {N_HIDDEN_TRF}')
        print(f'\tn_layer: {N_LAYER_TRF}')
        print(f'\tn_head: {N_HEAD}')
        print(f'\tn_positions: {MAX_LENGTH}')
        print(f'\tlr: {LR}')
        print(f'\twd: {WD}')
        print(f'\tmax_epochs: {MAX_EPOCHS}')
        print(f'\tlog_freq: {LOG_FREQ}')
        print(f'\teval_every: {EVAL_EVERY}')
        train_model(
            grammar,
            train_data,
            val_data,
            hparams,
            this_experiment_dir = os.path.join(this_experiment_dir, 'trf'),
            verbose = VERBOSE
        )
    
    print(f'Deleting train data...')
    shutil.rmtree(os.path.join(this_experiment_dir, 'train'))
    print(f'Deleting val data...')
    shutil.rmtree(os.path.join(this_experiment_dir, 'val'))

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

    for grammar_args in product(*grid.values()):
        main(grammar_args, j)

import os
import json
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
DEBUG = False
PATIENCE = 5 # number of evals to wait before breaking if no appreciable change
TOLERANCE = 1e-3 # proportion of loss decrease equivalent to "no appreciable change"
MIN_EVALS = 200
VERBOSE = False
BATCH_SIZE = 32
N_HEAD = 4 if not DEBUG else 2 # ignored by LSTM
MAX_LENGTH = 128
MAX_EPOCHS = 20 if not DEBUG else 20
LR = 1e-3
WD = 1e-5
LOG_FREQ = 100 if not DEBUG else 16
EVAL_EVERY = 100 if not DEBUG else 64
NUM_SEQS_TRAIN = 32_000 if not DEBUG else 256
NUM_SEQS_VAL = 32_000 if not DEBUG else 256

# model-specific
N_EMBD_LSTM = 128 if not DEBUG else 64
N_EMBD_TRF = 256 if not DEBUG else 64

N_HIDDEN_LSTM = 256 if not DEBUG else 64
N_HIDDEN_TRF = 128 if not DEBUG else 64

N_LAYER_LSTM = 6 if not DEBUG else 3
N_LAYER_TRF = 4 if not DEBUG else 3

# formalism-specific
ngram_orders = [3, 4, 5]
pfsa_nums_states = [2, 4, 8, 16, 32, 64]
pcfg_nums_nts = [2, 4, 8, 16, 32, 64]

# constant over formalisms
default_grid = OrderedDict({
    'seed': [0, 1, 2],
    'num_symbols': [1_000, 5_000],
    'entropy': [0]
})

ngram_grid = default_grid.copy()
ngram_grid['order'] = ngram_orders

pfsa_grid = default_grid.copy()
pfsa_grid['num_states'] = pfsa_nums_states

pcfg_grid = default_grid.copy()
pcfg_grid['num_non_terminals'] = pcfg_nums_nts

def already_done():
    os.makedirs('experiments', exist_ok=True)
    experiments = [
        os.path.join('experiments', x)
        for x in os.listdir('experiments')
    ]
    
    exists_already = set()
    
    for experiment in experiments:
        
        lstm_hparams = os.path.join(experiment, 'lstm', 'hparams.json')
        if os.path.exists(lstm_hparams):
            lstm_grammar_str = json.load(open(lstm_hparams, 'r', encoding='utf-8'))['grammar_str']
            exists_already.add(('lstm', lstm_grammar_str))
            
        trf_hparams = os.path.join(experiment, 'trf', 'hparams.json')
        if os.path.exists(trf_hparams):
            trf_grammar_str = json.load(open(trf_hparams, 'r', encoding='utf-8'))['grammar_str']
            exists_already.add(('trf', trf_grammar_str))
            
    return exists_already

def main(grammar_args, j):
    
    if not os.path.exists('experiments'):
        this_experiment = '1'
    elif len(os.listdir('experiments')) == 0:
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
    
    finished = already_done()
    
    if ('lstm', grammar.file_name_convention) in finished:
        do_lstm = False
    else:
        do_lstm = True
        
    if ('trf', grammar.file_name_convention) in finished:
        do_trf = False
    else:
        do_trf = True
        
    if not do_lstm and not do_trf:
        return

    # grammar = grammar.to(DEVICE)
    # print(f'Optimizing {grammar} on {grammar.device} to have entropy {entropy}...', flush=True)
    # if not grammar.optimize(H_t=entropy, do_logging=True):
    #     print(f'Optimization failed. Consider retrying.', flush=True)
    #     if not os.path.exists('failed_opt.tsv'):
    #         open('failed_opt.tsv', 'w+', encoding='utf-8')
    #     with open('failed_opt.tsv', 'r', encoding='utf-8') as f:
    #         line = f'{grammar}\t{entropy}'
    #         if line not in f.read().splitlines():
    #             f.close()
    #             with open('failed_opt.tsv', 'a', encoding='utf-8') as g:
    #                 g.write(line + '\n')
    # grammar = grammar.to('cpu')
    
    # print(f'Target entropy: {entropy}.', flush=True)
    ge = grammar.entropy().item()
    print(f'Grammar entropy: {ge}', flush=True)
    # print(f'Diff: {abs(ge - entropy)}', flush=True)
    
    print(f'Generating {NUM_SEQS_TRAIN:,} sequences with {grammar}...', flush=True)
    train_data = dataset_type(
        grammar,
        num_seqs=NUM_SEQS_TRAIN,
        max_length=MAX_LENGTH,
        do_logging=False,
        data_dir=os.path.join(this_experiment_dir, 'train')
    )
    print(f'Generating {NUM_SEQS_VAL:,} sequences with {grammar}...', flush=True)
    val_data = dataset_type(
        grammar,
        num_seqs=NUM_SEQS_VAL,
        max_length=MAX_LENGTH,
        do_logging=False,
        data_dir=os.path.join(this_experiment_dir, 'val')
    )
    
    print(f'Computing train excess entropy...', flush=True)
    train_ee = train_data.excess_entropy()
    print(f'Computing val excess entropy...', flush=True)
    val_ee = val_data.excess_entropy()
    print(f'Computing basic stats...', flush=True)
    hparams = {
        'grammar_type': grammar.formalism,
        'grammar_seed': grammar.seed,
        'grammar_num_symbols': grammar.num_symbols,
        'grammar_str': grammar.file_name_convention,
        'grammar_formalism_arg': formalism_arg,
        # 'grammar_target_entropy': entropy,
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
        'tolerance': TOLERANCE,
        'min_evals': MIN_EVALS
    }

    if do_lstm:
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
            hparams,
            this_experiment_dir = os.path.join(this_experiment_dir, 'lstm'),
            verbose = VERBOSE
        )
    
    if do_trf:
        hparams['model_type'] = 'trf'
        hparams['n_embd'] = N_EMBD_TRF
        hparams['n_hidden'] = N_HIDDEN_TRF
        hparams['n_layer'] = N_LAYER_TRF
        
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
            hparams,
            this_experiment_dir = os.path.join(this_experiment_dir, 'trf'),
            verbose = VERBOSE
        )
    
    if os.path.exists(os.path.join(this_experiment_dir, 'train')):
        print(f'Deleting train data...', flush=True)
        shutil.rmtree(os.path.join(this_experiment_dir, 'train'))
    if os.path.exists(os.path.join(this_experiment_dir, 'val')):
        print(f'Deleting val data...', flush=True)
        shutil.rmtree(os.path.join(this_experiment_dir, 'val'))
    
    del grammar, train_data, val_data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-j',
        type=int,
        required=True,
        help='0 for n-grams, 1 for PFSAs, 2 for PCFGs.'
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
        main(grammar_args, j)

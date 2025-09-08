import os
from collections import OrderedDict
from itertools import product

import torch

from pcfg import PCFG

os.makedirs('grammar_matrices', exist_ok=True)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

grid = OrderedDict({
    'num_non_terminals': [10, 50, 100],
    'num_terminals':     [10, 50, 100, 500, 1000, 5000, 10000],
    'entropy':           [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    'seed':              [0, 42, 69, 420, 2025]
})

for vals in product(*grid.values()):

    combo = dict(zip(grid.keys(), vals))

    name = ''
    for k, v in combo.items():
        name += str(k) + '_' + str(v) + '_'
    name = name[:-1] + '.pt'

    path = os.path.join(
        'grammar_matrices',
        name
    )

    if os.path.exists(path):
        print(f'{path} exists. Skipping.')
        continue
        

    print(f'Hyperparameters: {combo}')

    grammar = PCFG(
        num_non_terminals=combo['num_non_terminals'],
        num_terminals=combo['num_terminals'],
        seed=combo['seed']
    )
    grammar.to(device)
    grammar.optimize(float(combo['entropy']))

    grammar.save(path)

    print(f'Wrote {path}')
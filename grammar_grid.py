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
    'num_non_terminals': [8, 32],
    'num_terminals':     [4_096, 16_384, 65_536],
    'entropy':           [8, 32, 128],
    'seed':              [0]
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
        g = PCFG(rules=torch.load(path))
        rho = g._char_matrix_rho().item()
        if rho >= 1.0:
            print(f'{path} exists. Skipping.')
            continue
        else:
            print(f'{path} exists with bad rho. Re-trying.')

    print(f'Hyperparameters: {combo}')

    grammar = PCFG(
        num_non_terminals=combo['num_non_terminals'],
        num_terminals=combo['num_terminals'],
        seed=combo['seed']
    )
    # grammar.to(device)
    success = grammar.optimize(float(combo['entropy']))

    if abs(grammar.entropy() - float(combo['entropy'])) > 5:
        print(f'{combo} failed.')
    elif success:
        grammar.save(path)
        print(f'Wrote {path}')
    else:
        print(f'{combo} failed.')
import glob
import json

from pcfg import PCFG

for grammar in glob.glob('experiments/*/lstm/hparams.json'):
    hparams = json.load(open(grammar, 'r', encoding='utf-8'))
    if 'good_turing' in hparams:
        continue
    if hparams['grammar_type'] != 'pcfg':
        continue
    
    if 'var' in hparams:
        var = hparams['var']
    else:
        var = 1.0
    
    g = PCFG(
        seed=hparams['grammar_seed'],
        num_symbols=hparams['grammar_num_symbols'],
        num_non_terminals=hparams['grammar_formalism_arg'],
        var=var
    )
    
    hparams['good_turing'] = g.adaptive_good_turing_entropy()
    
    json.dump(hparams, open(grammar, 'w+', encoding='utf-8'), indent=4)

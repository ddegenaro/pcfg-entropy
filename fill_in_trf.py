import glob, json

for grammar in glob.glob('experiments/*/trf/hparams.json'):
    hparams_trf = json.load(open(grammar, 'r', encoding='utf-8'))
    
    if hparams_trf['grammar_type'] != 'pcfg':
        continue
    
    if 'good_turing' not in hparams_trf:
        hparams_lstm = json.load(open(grammar.replace('trf', 'lstm'), 'r', encoding='utf-8'))
        hparams_trf['good_turing'] = hparams_lstm['good_turing']
        json.dump(hparams_trf, open(grammar, 'w+', encoding='utf-8'), indent=4)

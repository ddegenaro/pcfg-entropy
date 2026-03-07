import glob, json

for grammar in glob.glob('experiments/*/*/hparams.json'):
    hparams = json.load(open(grammar, 'r', encoding='utf-8'))
    
    if 'var' not in hparams:
        hparams['var'] = 1.0
        json.dump(hparams, open(grammar, 'w+', encoding='utf-8'), indent=4)

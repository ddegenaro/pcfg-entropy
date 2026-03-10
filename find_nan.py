import glob
import pandas as pd
import json

c = 0

files = glob.glob('experiments/*/*/metrics.tsv')

for log_tsv in files:
    df = pd.read_csv(log_tsv, sep='\t')
    s = df['ce'].isna().sum()
    if s > 0:
        j = json.load(open(log_tsv.replace('metrics.tsv', 'hparams.json')))
        var = j['var']
        arch = 'lstm' if 'lstm' in log_tsv else 'trf'
        grammar_type = j['grammar_type']
        num_symbols = j['grammar_num_symbols']
        nts = j['grammar_formalism_arg']
        seed = j['grammar_seed']
        print(arch, grammar_type, var, num_symbols, nts, seed)
        c += 1
        
print(f'nan found in {c}/{len(files)} logs')
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
        print(log_tsv, f'var={j['var']}', f'{round(100 * s / len(df))}')
        c += 1
        
print(f'nan found in {c}/{len(files)} logs')
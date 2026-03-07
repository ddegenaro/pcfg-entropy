import glob
import pandas as pd
import numpy as np

for log_tsv in glob.glob('experiments/*/*/metrics.tsv'):
    df = pd.read_csv(log_tsv, sep='\t')
    s = df['ce'].isna().sum()
    if s > 0:
        print(log_tsv, s, '/', len(df))
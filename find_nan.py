import glob
import pandas as pd
import numpy as np

for log_tsv in glob.glob('experiments/*/*/metrics.tsv'):
    df = pd.read_csv(log_tsv, sep='\t')
    if 'nan' in df['ce']:
        print(log_tsv)
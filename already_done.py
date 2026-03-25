import json
import glob

already_done_lstm = set()
already_done_trf = set()

lstm_jsons = glob.glob('experiments/*/lstm/hparams.json')
trf_jsons = glob.glob('experiments/*/trf/hparams.json')

for j in lstm_jsons:
    already_done_lstm.add(json.load(open(j))['grammar_str'])
    
for j in trf_jsons:
    already_done_trf.add(json.load(open(j))['grammar_str'])
    
print(len(already_done_lstm), len(already_done_trf))
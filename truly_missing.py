import glob, json

hparams_lstms = glob.glob('experiments/*/lstm/hparams.json')
hparams_trfs = glob.glob('experiments/*/trf/hparams.json')

for line in open('missing.txt', 'r', encoding='utf-8'):
    data = eval(line.strip())
    if data[0] == 'lstm':
        files = hparams_lstms
    elif data[0] == 'trf':
        files = hparams_trfs
    else:
        print(data[0], 'what???')
        
    flag = False
        
    for file in files:
        j = json.load(open(file, 'r', encoding='utf-8'))
        
        if j['grammar_str'] == data[1] and j['var'] == data[2]:
            flag = True
            break
        
    if not flag:
        print(data[1], data[2])
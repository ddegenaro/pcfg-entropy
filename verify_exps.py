import glob, json, os

experiments_lstm = set(glob.glob('experiments/*/lstm/hparams.json'))
experiments_trf = set(glob.glob('experiments/*/trf/hparams.json'))

def already_done():
    os.makedirs('experiments', exist_ok=True)
    experiments = [
        os.path.join('experiments', x)
        for x in os.listdir('experiments')
    ]
    
    exists_already = set()
    
    for experiment in experiments:
        
        lstm_hparams = os.path.join(experiment, 'lstm', 'hparams.json')
        if os.path.exists(lstm_hparams):
            lstm_grammar_str = json.load(open(lstm_hparams, 'r', encoding='utf-8'))['grammar_str']
            var = lstm_hparams['var'] if 'var' in lstm_hparams else 1.0
            exists_already.add(('lstm', lstm_grammar_str, var))
            
        trf_hparams = os.path.join(experiment, 'trf', 'hparams.json')
        if os.path.exists(trf_hparams):
            trf_grammar_str = json.load(open(trf_hparams, 'r', encoding='utf-8'))['grammar_str']
            var = trf_hparams['var'] if 'var' in trf_hparams else 1.0
            exists_already.add(('trf', trf_grammar_str, var))
            
    return exists_already

for experiment_lstm in experiments_lstm:
    experiment_trf = experiment_lstm.replace('lstm', 'trf')
    assert experiment_trf in experiments_trf

    lstm_json = json.load(open(experiment_lstm))
    trf_json = json.load(open(experiment_trf))
    
    for key in lstm_json:
        if key not in ('n_embd', 'n_hidden', 'n_layer', 'model_type', 'param_count'):
            if type(lstm_json[key]) == dict:
                for second_key in lstm_json[key]:
                    assert lstm_json[key][second_key] == trf_json[key][second_key]
            else:
                assert lstm_json[key] == trf_json[key], key
                
    if lstm_json['grammar_type'] == 'pcfg':
        assert 'good_turing' in lstm_json, experiment_lstm

opts = {
    'formalism_arg': [2, 4, 8, 16, 32, 64],
    'seed': [0, 1, 2],
    'num_symbols': [1_000, 5_000],
    'var': [1.0, 4.0]
}

ad = already_done()

for formalism_arg in opts['formalism_arg']:
    for seed in opts['seed']:
        for num_symbols in opts['num_symbols']:
            for var in opts['var']:
                pfsa_str = f'pfsa_seed_{seed}_symbols_{num_symbols}_states_{formalism_arg}'
                pcfg_str = f'pcfg_seed_{seed}_symbols_{num_symbols}_nts_{formalism_arg}'
                
                if ('lstm', pfsa_str, var) not in ad:
                    print('lstm', pfsa_str, var)
                if ('trf', pfsa_str, var) not in ad:
                    print('trf', pfsa_str, var)
                if ('lstm', pcfg_str, var) not in ad:
                    print('lstm', pcfg_str, var)
                if ('trf', pcfg_str, var) not in ad:
                    print('trf', pcfg_str, var)
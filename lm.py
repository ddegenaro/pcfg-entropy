import os
import json

import torch
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel

from lstm import LSTM
from utils import Grammar, SequenceDataset
from metrics import both_metrics

def create_model_and_optimizer(
    grammar: Grammar,
    n_embd: int,
    n_hidden: int,
    n_layer: int,
    n_head: int,
    n_positions: int,
    lr: int,
    wd: int,
    model_type: str
):
    
    if model_type == 'trf':
        model = GPT2LMHeadModel(
            config=GPT2Config(
                vocab_size=grammar.num_symbols + 3, # EOS, PAD
                pad_token_id=grammar.num_symbols + 2,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_positions=n_positions
            )
        )
    elif model_type == 'lstm':
        model = LSTM(
            vocab_size=grammar.num_symbols + 3, # EOS, PAD
            pad_token_id = grammar.num_symbols + 2,
            n_embd=n_embd,
            n_hidden=n_hidden,
            n_layer=n_layer
        )
    else:
        raise ValueError('model_type must be either "trf" or "lstm"')

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Training {model_type} with {param_count:,} trainable parameters.', flush=True)

    return (model, optimizer, param_count)

def do_log(msg: str):
    print(msg, flush=True)

def train_epoch(
    grammar: Grammar,
    train_data: SequenceDataset,
    model: torch.nn.Module,
    optimizer: AdamW,
    epoch: int,
    train_losses: list[float],
    train_tokens: list[int],
    log_freq: int,
    eval_every: int,
    val_data: SequenceDataset,
    rhos: dict[int, float],
    ces: dict[int, float],
    p_true: list[float]
):
    
    model.train()

    running_total_train_loss = sum(l * t for l, t in zip(train_losses, train_tokens))
    running_total_train_tokens = sum(train_tokens)

    do_log('-' * 100)
    do_log(f'Begin training epoch {epoch}.')
    do_log('-' * 100)

    for i, batch in enumerate(train_data):
        
        batch = grammar.batch_tokenize(batch, return_tensors='pt')
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

        loss = outputs.loss.item()
        tokens = batch['attention_mask'].sum()
        running_total_train_loss += loss
        running_total_train_tokens += tokens
        train_losses.append(loss)
        train_tokens.append(tokens)
        avg_loss = running_total_train_loss / running_total_train_tokens
        step = i + 1

        if step % log_freq == 0:
            msg = f'Epoch: {epoch:03d} - Step: {step:05d} - Loss: {loss:.4f} - Avg: {avg_loss:.4f}'
            do_log(msg)

        if step % eval_every == 0:
            rho, ce = val_epoch(
                val_data,
                model,
                step,
                p_true
            )
    
            rhos[step] = rho
            ces[step] = ce

def val_epoch(
    val_data: SequenceDataset,
    model: torch.nn.Module,
    step: int,
    p_true: list[float]
):

    do_log('-' * 100)
    do_log(f'Begin eval after {step} steps.')
    do_log('-' * 100)

    return both_metrics(val_data, model, p_true)

def train_model(
    grammar: Grammar,
    train_data: SequenceDataset,
    val_data: SequenceDataset,
    n_embd: int = 256,
    n_hidden: int = 128,
    n_layer: int = 6,
    n_head: int = 4,
    n_positions: int = 256,
    lr: int = 1e-3,
    wd: int = 1e-5,
    max_epochs: int = 20,
    log_freq: int = 100,
    eval_every: int = 1000,
    trf_or_lstm: str = 'trf'
):
    
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
        this_experiment = '1'
    else:
        experiments = [int(x) for x in os.listdir('experiments')]
        this_experiment = max(experiments) + 1
    this_experiment_dir = os.path.join('experiments', this_experiment)
    os.makedirs(this_experiment_dir)
    
    train_losses = []
    train_tokens = []
    rhos = {}
    ces = {}
    
    model, optimizer, param_count = create_model_and_optimizer(
        grammar,
        n_embd=n_embd,
        n_hidden=n_hidden,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=n_positions,
        lr=lr,
        wd=wd,
        model_type=trf_or_lstm
    )

    hparams = {
        'grammar_type': grammar.formalism,
        'grammar_seed': grammar.seed,
        'grammar_num_symbols': grammar.num_symbols,
        'grammar_str': grammar.file_name_convention,
        'train_data_stats': train_data.basic_stats(),
        'train_data_ee': train_data.excess_entropy(),
        'val_data_stats': val_data.basic_stats(),
        'val_data_ee': val_data.excess_entropy(), 
        'n_embd': n_embd,
        'n_hidden': n_hidden,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_positions': n_positions,
        'lr': lr,
        'wd': wd,
        'max_epochs': max_epochs,
        'log_freq': log_freq,
        'model_type': trf_or_lstm,
        'param_count': param_count
    }

    with open(os.path.join(this_experiment_dir, 'hparams.json'), 'w+', encoding='utf-8') as f:
        json.dump(hparams, f, indent=4)

    p_true = [grammar.p_seq(seq).item() for seq in val_data]

    for epoch in max_epochs:
        train_epoch(
            grammar=grammar,
            train_data=train_data,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_losses=train_losses,
            train_tokens=train_tokens,
            log_freq=log_freq,
            eval_every=eval_every, # evaluate every X steps
            val_data=val_data,
            rhos=rhos,
            ces=ces,
            p_true=p_true
        )

    with open(os.path.join(this_experiment_dir, 'train_losses.tsv'), 'w+', encoding='utf-8') as f:
        f.write('avg_loss\ttokens\n')
        for i in range(len(train_losses)):
            f.write(f'{train_losses[i]}\t{train_tokens[i]}\n')

    with open(os.path.join(this_experiment_dir, 'metrics.tsv'), 'w+', encoding='utf-8') as f:
        f.write('step\trho\tce\n')
        for step in rhos.keys():
            f.write(f'{step}\t{rhos[step]}\t{ces[step]}\n')
import os
import json
from typing import Union

from tqdm import tqdm
import torch
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel

from lstm import LSTM
from utils import Grammar, SequenceDataset, SequenceDataLoader
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
) -> Union[tuple[LSTM, AdamW, int], tuple[GPT2LMHeadModel, AdamW, int]]:
    
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
    train_data_loader: SequenceDataLoader,
    model: torch.nn.Module,
    optimizer: AdamW,
    epoch: int,
    train_losses: list[float],
    train_tokens: list[int],
    log_freq: int,
    eval_every: int,
    val_data_loader: SequenceDataLoader,
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

    step = len(train_data_loader) * epoch

    for batch in tqdm(train_data_loader, total=len(train_data_loader)):

        batch['labels'] = batch['input_ids']
        
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
        step += 1

        if step % log_freq == 0:
            msg = f'Epoch: {epoch:03d} - Step: {step:05d} - Loss: {loss:.4f} - Avg/tok: {avg_loss:.4f}'
            do_log(msg)

        if step % eval_every == 0:
            rho, ce = val_epoch(
                val_data_loader=val_data_loader,
                model=model,
                step=step,
                p_true=p_true
            )
    
            rhos[step] = rho
            ces[step] = ce

def val_epoch(
    val_data_loader: SequenceDataLoader,
    model: torch.nn.Module,
    step: int,
    p_true: list[float]
):

    do_log('-' * 100)
    do_log(f'Begin eval after {step} steps.')
    do_log('-' * 100)

    return both_metrics(
        val_data_loader=val_data_loader,
        model=model,
        p_true=p_true
    )

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
    trf_or_lstm: str = 'trf',
    batch_size: int = 32,
    is_test_run: bool = False
):
    
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
        this_experiment = '1'
    else:
        experiments = [int(x.replace('_test', '')) for x in os.listdir('experiments')]
        this_experiment = str(max(experiments) + 1)
    if is_test_run:
        this_experiment += '_test'
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

    print(f'Building train dataloader...')
    train_data_loader = SequenceDataLoader(
        ds=train_data,
        batch_size=batch_size,
        shuffle=True,
        max_length=n_positions
    )

    print(f'Building val dataloader...')
    val_data_loader = SequenceDataLoader(
        ds=val_data,
        batch_size=batch_size,
        shuffle=False,
        max_length=n_positions
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

    hparams_loc = os.path.join(this_experiment_dir, 'hparams.json')
    print(f'Writing hparams to {hparams_loc}')
    with open(hparams_loc, 'w+', encoding='utf-8') as f:
        json.dump(hparams, f, indent=4)

    print(f'Computing p_true...')
    p_true = [grammar.p_seq(seq).item() for seq in tqdm(val_data, total=len(val_data))]

    for epoch in range(max_epochs):

        torch.manual_seed(
            torch.randint(
                low=0, high=max_epochs**2, size=(1,),
                generator=grammar.cpu_generator
            ).item()
        )

        train_epoch(
            train_data_loader=train_data_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_losses=train_losses,
            train_tokens=train_tokens,
            log_freq=log_freq,
            eval_every=eval_every, # evaluate every X steps
            val_data_loader=val_data_loader,
            rhos=rhos,
            ces=ces,
            p_true=p_true
        )

    with open(os.path.join(this_experiment_dir, 'train_losses.tsv'), 'w+', encoding='utf-8') as f:
        f.write('avg_loss\ttokens\n')
        for i in range(len(train_losses)):
            f.write(f'{train_losses[i]}\t{train_tokens[i]}\n')

    with open(os.path.join(this_experiment_dir, 'metrics.tsv'), 'w+', encoding='utf-8') as f:
        f.write('step\trho\trho_pval\tce\n')
        for step in rhos.keys():
            f.write(f'{step}\t{rhos[step].statistic}\t{rhos[step.pvalue]}\t{ces[step]}\n')
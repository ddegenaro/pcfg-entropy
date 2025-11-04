from typing import Union

import torch
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel

from lm import LSTM
from utils import Grammar, SequenceDataset

def create_model_and_optimizer(
    grammar: Grammar,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 4,
    n_positions: int = 1024,
    lr: int = 1e-3,
    wd: int = 1e-5,
    model_type: str = 'trf'
):
    
    if model_type == 'trf':
        model = GPT2LMHeadModel(
            config=GPT2Config(
                vocab_size=grammar.num_symbols + 2, # EOS, PAD
                eos_token_id=grammar.num_symbols,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_positions=n_positions
            )
        )
    else:
        model = LSTM(
            vocab_size=grammar.num_symbols + 2, # EOS, PAD
            eos_token_id = grammar.num_symbols,
            n_embd=n_embd,
            n_layer=n_layer,
            n_positions=n_positions
        )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    return (model, optimizer)

def do_log(msg: str):
    print(msg, flush=True)

def train_epoch(
    grammar: Grammar,
    train_data: SequenceDataset,
    model: torch.nn.Module,
    optimizer: AdamW,
    epoch: int,
    train_losses: list[float],
    log_freq: int,
    eval_every: int,
    val_data: SequenceDataset,
    val_losses: list[float]
):
    
    model.train()

    running_total_train_loss = sum(train_losses)
    running_total_train_steps = len(train_losses)

    do_log('-' * 100)
    do_log(
        f'Begin training epoch {epoch}.'
    )
    do_log('-' * 100)

    for i, batch in enumerate(train_data):
        
        batch = grammar.batch_tokenize(batch, return_tensors='pt')
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

        loss = outputs.loss.item()
        train_losses.append(loss)
        running_total_train_loss += loss
        running_total_train_steps += 1
        avg_loss = running_total_train_loss / running_total_train_steps
        step = i + 1

        if step % log_freq == 0:
            msg = f'Epoch: {epoch:03d} - Step: {step:05d} - Loss: {loss:.4f} - Avg: {avg_loss:.4f}'
            do_log(msg)

        if step % eval_every == 0:
            val_epoch(
                grammar,
                val_data,
                model,
                epoch,
                step,
                val_losses,
                log_freq
            )

def val_epoch(
    grammar: Grammar,
    val_data: SequenceDataset,
    model: torch.nn.Module,
    epoch: int,
    step: int,
    val_losses: list[float],
    log_freq: int
):
    
    model.eval()

    running_total_val_loss = sum(val_losses)
    running_total_val_steps = len(val_losses)

    do_log('-' * 100)
    do_log(
        f'Begin eval after {step} steps.'
    )
    do_log('-' * 100)

    with torch.no_grad():

        for i, batch in enumerate(val_data):
        
            batch = grammar.batch_tokenize(batch, return_tensors='pt')
            outputs = model(**batch)

            loss = outputs.loss.item()
            val_losses.append(loss)
            running_total_val_loss += loss
            running_total_val_steps += 1
            avg_loss = running_total_val_loss / running_total_val_steps
            step = i + 1

            if step % log_freq == 0:
                msg = f'Epoch: {epoch:03d} - Step: {step:05d} - Loss: {loss:.4f} - Avg: {avg_loss:.4f}'
                do_log(msg)

def train_model(
    grammar: Grammar,
    train_data: SequenceDataset,
    val_data: SequenceDataset,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 4,
    n_positions: int = 256,
    lr: int = 1e-3,
    wd: int = 1e-5,
    max_epochs: int = 20,
    log_freq: int = 1000,
    trf_or_lstm: str = 'trf'
):
    
    train_losses = []
    val_losses = []
    
    model, optimizer = create_model_and_optimizer(
        grammar,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=n_positions,
        lr=lr,
        wd=wd,
        model_type=trf_or_lstm
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Training LM with {param_count:,} trainable parameters.', flush=True)

    hparams = {
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_positions': n_positions,
        'lr': lr,
        'wd': wd,
        'max_epochs': max_epochs,
        'param_count': param_count
    }

    for epoch in max_epochs:
        train_epoch(
            grammar=grammar,
            train_data=train_data,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_losses=train_losses,
            log_freq=log_freq,
            eval_every=1_000, # evaluate every X steps
            val_data=val_data,
            val_losses=val_losses
        )

    return train_losses, val_losses, model, optimizer, hparams
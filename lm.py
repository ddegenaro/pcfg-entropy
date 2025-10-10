import torch
from torch.optim import AdamW
from transformers import GPT2Config, GPT2LMHeadModel

from pcfg import PCFG, PCFGDataset

def create_model_and_optimizer(
    grammar: PCFG,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 4,
    n_positions: int = 1024,
    lr: int = 1e-3,
    wd: int = 1e-5
):
    model = GPT2LMHeadModel(
        config=GPT2Config(
            vocab_size=grammar.num_terminals + 1,
            eos_token_id=grammar.num_terminals,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=n_positions
        )
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
    grammar: PCFG,
    train_data: PCFGDataset,
    model: GPT2LMHeadModel,
    optimizer: AdamW,
    epoch: int,
    train_losses: list[float],
    log_freq: int
):
    
    model.train()

    running_total_train_loss = sum(train_losses)
    running_total_train_steps = len(train_losses)

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

def val_epoch(
    grammar: PCFG,
    val_data: PCFGDataset,
    model: GPT2LMHeadModel,
    epoch: int,
    val_losses: list[float],
    log_freq: int
):
    
    model.eval()

    running_total_val_loss = sum(val_losses)
    running_total_val_steps = len(val_losses)

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
    grammar: PCFG,
    train_data: PCFGDataset,
    val_data: PCFGDataset,
    n_embd: int = 256,
    n_layer: int = 6,
    n_head: int = 4,
    n_positions: int = 1024,
    lr: int = 1e-3,
    wd: int = 1e-5,
    max_epochs: int = 20,
    log_freq: int = 1000
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
        wd=wd
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Training LM with {param_count:,} trainable parameters.', flush=True)

    for epoch in max_epochs:
        train_epoch(
            grammar=grammar,
            train_data=train_data,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_losses=train_losses,
            log_freq=log_freq
        )
        val_epoch(
            grammar=grammar,
            val_data=val_data,
            model=model,
            epoch=epoch,
            val_losses=val_losses,
            log_freq=log_freq
        )
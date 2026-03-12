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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                vocab_size=grammar.num_symbols + 3, # BOS, EOS, PAD
                bos_token_id=grammar.num_symbols,
                eos_token_id=grammar.num_symbols + 1,
                pad_token_id=grammar.num_symbols + 2,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                n_positions=n_positions + 1 # for start token
            )
        ).to(device=DEVICE, dtype=torch.bfloat16)
        model.bos_token_id = grammar.num_symbols
        model.eos_token_id = grammar.num_symbols + 1
        model.pad_token_id = grammar.num_symbols + 2
    elif model_type == 'lstm':
        model = LSTM(
            vocab_size=grammar.num_symbols + 3, # BOS, EOS, PAD
            bos_token_id=grammar.num_symbols,
            eos_token_id=grammar.num_symbols + 1,
            pad_token_id = grammar.num_symbols + 2,
            n_embd=n_embd,
            n_hidden=n_hidden,
            n_layer=n_layer
        ).to(device=DEVICE, dtype=torch.bfloat16)
    else:
        raise ValueError('model_type must be either "trf" or "lstm"')

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f'Training {model_type} on {DEVICE} with {param_count:,} trainable parameters.', flush=True)
    
    return (model, optimizer, param_count)



def train_epoch(
    train_data_loader: SequenceDataLoader,
    model: torch.nn.Module,
    optimizer: AdamW,
    epoch: int,
    log_freq: int,
    eval_every: int,
    val_data_loader: SequenceDataLoader,
    log_p_true_by_len: dict[int, list[float]],
    this_experiment_dir: str,
    verbose: bool,
    all_train_losses: list[float],
    all_train_tokens: list[int],
    last_k_ces: list[float],
    patience: int,
    tol: float,
    min_evals: int
):
    
    os.makedirs
    
    model.train()

    running_total_train_loss = sum(l * t for l, t in zip(all_train_losses, all_train_tokens))
    running_total_train_tokens = sum(all_train_tokens)

    print('-' * 100, flush=True)
    print(f'Begin training epoch {epoch+1}.', flush=True)
    print('-' * 100, flush=True)

    step = len(train_data_loader) * epoch

    if verbose:
        iterable = tqdm(train_data_loader, total=len(train_data_loader))
    else:
        iterable = train_data_loader

    train_losses = []
    train_tokens = []

    for batch in iterable:
        
        # prepend start token to each sequence
        batch['input_ids'] = torch.hstack((
            torch.full((batch['input_ids'].shape[0], 1), model.bos_token_id),
            batch['input_ids']
        ))
        
        # put 1's there for attention mask, ok to attend to BOS
        batch['attention_mask'] = torch.hstack((
            torch.full((batch['attention_mask'].shape[0], 1), 1),
            batch['attention_mask']
        ))

        batch['labels'] = batch['input_ids']

        for key in batch:
            batch[key] = batch[key].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()

        loss = outputs.loss.item()
        tokens = batch['attention_mask'].sum().item()
        running_total_train_loss += loss * tokens # this average loss, times this many tokens
        running_total_train_tokens += tokens
        train_losses.append(loss)
        train_tokens.append(tokens)
        avg_loss = running_total_train_loss / running_total_train_tokens
        step += 1

        if step % log_freq == 0:
            msg = f'Epoch: {epoch+1:03d} - Step: {step:05d} - Loss: {loss:.4f} - Avg/tok: {avg_loss:.4f}'
            print(msg, flush=True)

        if step % eval_every == 0:
            ce = val_epoch(
                val_data_loader=val_data_loader,
                model=model,
                step=step,
                log_p_true_by_len=log_p_true_by_len,
                this_experiment_dir=this_experiment_dir,
                verbose=verbose
            )
            
            with open(
                os.path.join(this_experiment_dir, 'train_losses.tsv'), 'a', encoding='utf-8'
            ) as f:
                for i in range(len(train_losses)):
                    f.write(f'{train_losses[i]}\t{train_tokens[i]}\n')
                    
            all_train_losses.extend(train_losses)
            all_train_tokens.extend(train_tokens)
                    
            train_losses = []
            train_tokens = []
            
            if round(step / eval_every) < min_evals:
                continue
            else:
                if len(last_k_ces) == patience:
                    del last_k_ces[0]
                last_k_ces.append(ce)
                
                if len(last_k_ces) == patience:
                    
                    no_change_flags = []
                    loss_growing_flags = []
                    
                    for i in range(1, len(last_k_ces)):
                        if abs(last_k_ces[i-1] - last_k_ces[i]) < last_k_ces[i-1] * tol:
                            no_change_flags.append(True)
                        else:
                            no_change_flags.append(False)
                        
                        if last_k_ces[i] > last_k_ces[i-1]:
                            loss_growing_flags.append(True)
                        else:
                            loss_growing_flags.append(False)
                            
                    if all(no_change_flags):
                        return 'end'
                    
                    if all(loss_growing_flags):
                        return 'end'

def val_epoch(
    val_data_loader: SequenceDataLoader,
    model: torch.nn.Module,
    step: int,
    log_p_true_by_len: dict[int, list[float]],
    this_experiment_dir: str,
    verbose: bool
):

    print('-' * 100, flush=True)
    print(f'Begin eval after {step} steps.', flush=True)
    print('-' * 100, flush=True)

    count_by_len, spearman_by_len, pvalue_by_len, ce = both_metrics(
        val_data_loader=val_data_loader,
        model=model,
        log_p_true_by_len=log_p_true_by_len,
        device=DEVICE,
        verbose=verbose
    )
    
    seq_lens = sorted(list(spearman_by_len.keys()))
    
    spearman_weighted_avg = sum([
        spearman_by_len[seq_len] * count_by_len[seq_len] for seq_len in seq_lens
    ]) / sum([count_by_len[seq_len] for seq_len in seq_lens])
    
    with open(os.path.join(this_experiment_dir, 'metrics.tsv'), 'a', encoding='utf-8') as f:
        f.write(f'{step}\t{spearman_weighted_avg}\t{sum(pvalue_by_len.values())}\t{ce}\n')
    
    with open(os.path.join(this_experiment_dir, 'length_wise_metrics.tsv'), 'a', encoding='utf-8') as f:
        for seq_len in seq_lens:
            f.write(f'{step}\t{seq_len}\t{spearman_by_len[seq_len]}\t{pvalue_by_len[seq_len]}\n')
        
    return ce

def train_model(
    grammar: Grammar,
    train_data: SequenceDataset,
    val_data: SequenceDataset,
    hparams: dict,
    this_experiment_dir: str = '.',
    verbose: bool = False
):
    
    os.makedirs(this_experiment_dir, exist_ok=True)
    
    model, optimizer, param_count = create_model_and_optimizer(
        grammar,
        n_embd=hparams['n_embd'],
        n_hidden=hparams['n_hidden'],
        n_layer=hparams['n_layer'],
        n_head=hparams['n_head'],
        n_positions=hparams['n_positions'],
        lr=hparams['lr'],
        wd=hparams['wd'],
        model_type=hparams['model_type']
    )

    print(f'Building train dataloader...', flush=True)
    train_data_loader = SequenceDataLoader(
        ds=train_data,
        batch_size=hparams['batch_size'],
        shuffle=True,
        max_length=hparams['n_positions']
    )

    print(f'Building val dataloader...', flush=True)
    val_data_loader = SequenceDataLoader(
        ds=val_data,
        batch_size=hparams['batch_size'],
        shuffle=False,
        max_length=hparams['n_positions']
    )

    hparams['param_count'] = param_count

    hparams_loc = os.path.join(this_experiment_dir, 'hparams.json')
    print(f'Writing hparams to {hparams_loc}', flush=True)
    with open(hparams_loc, 'w+', encoding='utf-8') as f:
        json.dump(hparams, f, indent=4)

    print(f'Computing log_p_true_by_len...', flush=True)
    if verbose:
        log_p_true = [
            grammar.p_seq(seq).log().item()
            for seq in tqdm(val_data, total=len(val_data))
        ]
        lens = [
            len(seq) for seq in tqdm(val_data, total=len(val_data))
        ]
        log_p_true_by_len = {}
        for i in tqdm(range(len(lens)), total=len(lens)):
            if lens[i] not in log_p_true_by_len:
                log_p_true_by_len[lens[i]] = [log_p_true[i]]
            else:
                log_p_true_by_len[lens[i]].append(log_p_true[i])
    else:
        log_p_true = [grammar.p_seq(seq).log().item() for seq in val_data]
        lens = [len(seq) for seq in val_data]
        log_p_true_by_len = {}
        for i in range(len(lens)):
            if lens[i] not in log_p_true_by_len:
                log_p_true_by_len[lens[i]] = [log_p_true[i]]
            else:
                log_p_true_by_len[lens[i]].append(log_p_true[i])

    with open(os.path.join(this_experiment_dir, 'train_losses.tsv'), 'w+', encoding='utf-8') as f:
        f.write('avg_loss\ttokens\n')

    with open(os.path.join(this_experiment_dir, 'metrics.tsv'), 'w+', encoding='utf-8') as f:
        f.write('step\tspearman_weighted_avg\tsum_of_pvals\tce\n')
        
    with open(os.path.join(this_experiment_dir, 'length_wise_metrics.tsv'), 'w+', encoding='utf-8') as f:
        f.write(f'step\tseq_len\tspearman_by_len\tpval_by_len\n')
        
    all_train_losses = []
    all_train_tokens = []
    
    last_k_ces = []

    for epoch in range(hparams['max_epochs']):

        signal = train_epoch(
            train_data_loader=train_data_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            log_freq=hparams['log_freq'],
            eval_every=hparams['eval_every'], # evaluate every X steps
            val_data_loader=val_data_loader,
            log_p_true_by_len=log_p_true_by_len,
            this_experiment_dir=this_experiment_dir,
            verbose=verbose,
            all_train_losses=all_train_losses,
            all_train_tokens=all_train_tokens,
            last_k_ces=last_k_ces,
            patience=hparams['patience'],
            tol=hparams['tolerance'],
            min_evals=hparams['min_evals']
        )
        
        if signal == 'end':
            return
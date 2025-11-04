from time import time
from typing import Union

import torch
from torch import nn

from utils import Grammar, Sequence, SequenceDataset

class NGram(Grammar):

    def __init__(
        self,
        seed: int = 42,
        device: Union[str, torch.device] = 'cpu',
        chr_ord_offset: int = 97,
        from_file: str = None,
        num_symbols: int = 10,
        order: int = 5
    ):
        
        """
        Define an n-th order n-gram grammar with optimizable probabilities.

        Args:
            `seed`: `int` - random seed to generate reproducible initializations/optimization processes.
            `device`: `str` - PyTorch-compatible device name to run computations on GPUs when possible.
            `chr_ord_offset`: `int` - constant shift to Unicode representations of integer-valued terminal symbols.
            `from_file`: `str` - filepath specifying weights for this PFSA. `num_symbols` and `order` will be ignored.
            `num_symbols`: `int` - the number of symbols in the grammar. Used if `from_file` is None.
            `order`: `int` - the order (n) of the n-gram grammar - each token is sensitive to the preceding n-1 tokens.
        """
        
        # formalism-specific inits
        self.formalism = 'ngram'
        self.order: int = order
        assert self.order in {1, 2, 3, 4, 5} # probably dangerous to try 6, needs V^6 weights

        # load weights and validate
        super().__init__(
            seed,
            device,
            chr_ord_offset,
            from_file,
            num_symbols
        )
        self.file_name_convention = f'ngram_seed_{self.seed}_symbols_{self.num_symbols}_order_{self.order}'

        self.validate()

    def __repr__(self):
        return f'NGram(seed={self.seed}, num_symbols={self.num_symbols}, order={self.order})'

    def validate(self):
        super().validate()
        assert torch.isclose(self.probs['0'].sum(), torch.tensor(1., device=self.device))
        if self.order == 1:
            assert self.probs['0'].shape[0] == self.num_symbols + 1
        else:
            assert self.probs['0'].shape[0] == self.num_symbols
            for i in range(1, self.order - 1):
                assert self.probs[str(i)].shape[0] == self.num_symbols ** i
                assert self.probs[str(i)].shape[1] == self.num_symbols
                assert torch.allclose(self.probs[str(i)].sum(1), torch.tensor(1., device=self.device))
            assert self.probs[str(self.order - 1)].shape[0] == self.num_symbols  ** (self.order - 1)
            assert self.probs[str(self.order - 1)].shape[1] == self.num_symbols + 1
            assert torch.allclose(self.probs[str(self.order - 1)].sum(1), torch.tensor(1., device=self.device))
        assert [int(k) for k in self.probs.keys()] == [i for i in range(self.order)]

    def init_weights(self):
        self.probs: nn.ParameterDict[int, nn.Parameter] = nn.ParameterDict()
        if self.order == 1:
            self.probs['0'] = nn.Parameter(torch.randn( # unigrams no matter the order
                (self.num_symbols + 1,), # add 1 for stop
                device=self.device,
                generator=self.generator
            ).softmax(0))
        else:
            self.probs['0'] = nn.Parameter(torch.randn( # unigrams no matter the order
                (self.num_symbols,), # don't add 1 for stop - we don't stop with unigram
                device=self.device,
                generator=self.generator
            ).softmax(0))
            for i in range(1, self.order - 1): # i = 1...n-1 for n-grams
                self.probs[str(i)] = nn.Parameter(torch.randn(
                    (self.num_symbols ** i, self.num_symbols), # all possible contexts of length i -> any symbol
                    device=self.device,
                    generator=self.generator
                ).softmax(1))
            self.probs[str(self.order - 1)] = nn.Parameter(torch.randn(
                (self.num_symbols ** (self.order - 1), self.num_symbols + 1), # for context of length n-1, +1 because we might stop
                device=self.device,
                generator=self.generator
            ).softmax(1))

        super().init_weights() # turn off gradients

    def load(self, fp: str):
        self.probs = torch.load(fp)
        self.num_symbols = self.probs[1].shape[0] # length of unigram probability vector
        self.order = len(self.probs) # one entry per order, if order n then n entries for total info
    
    def save(self, fp: str):

        self.to('cpu')

        torch.save(self.probs, fp)
        print(f'Saved rules to {fp} successfully.')

        for k, v in self.probs.items():
            self.probs[k] = v.to(self.device)

    def p_seq(self, seq: Sequence):
        
        tokens = self.tokenize(seq)
        i = 1
        log_p = self.probs['0'][tokens[0]].log()

        # processes tokens 1, ..., self.order - 1 inclusive
        while str(i) in self.probs: # self.order not in self.probs
            idx = self._context_to_idx(tokens[:i])
            log_p += self.probs[str(i)][idx, tokens[i]].log()
            i += 1
        
        # processes self.order, ..., to the end
        for j in range(self.order, len(tokens)):
            idx = self._context_to_idx(tokens[j - self.order + 1:j])
            log_p += self.probs[str(self.order - 1)][idx, tokens[i]].log()
        
        return torch.exp(log_p)

    def _generate_one(self, max_length) -> Sequence:
        seq = [
            torch.multinomial(
                self.probs['0'],
                num_samples=1,
                generator=self.generator
            ).item()
        ]
        
        if self.order == 1 and seq[0] == self.num_symbols:
            return Sequence(self.untokenize(seq, False))

        # seq now has one symbol and we need to keep going
        i = 1
        while str(i) in self.probs:
            idx = torch.multinomial(
                self.probs[str(i)][self._context_to_idx(seq)], # sample from appropriate row given context
                num_samples=1,
                generator=self.generator
            ).item()
            seq.append(idx)
            i += 1
        
        def next_symbol():
            return torch.multinomial(
                self.probs[str(self.order - 1)][self._context_to_idx(seq[1-self.order:])],
                num_samples=1,
                generator=self.generator
            ).item()
        
        while len(seq) < max_length:
            # break if we generate stop symbol
            if seq[-1] == self.num_symbols:
                break
            seq.append(next_symbol())
        
        return Sequence(self.untokenize(seq, False))

    def entropy(self) -> torch.Tensor: # GENERATED WITH CLAUDE
        """
        Compute entropy over sequences by finding the stationary distribution
        of the n-gram model viewed as a Markov chain.
        
        For order n, contexts of length (n-1) are states. We compute the stationary
        distribution π over these states, then compute:
        H = -Σ_context π(context) * Σ_symbol P(symbol|context) * log P(symbol|context)
        """
        
        if self.order == 1:
            # For unigrams, entropy is just over the symbol distribution
            probs = self.probs['0']
            return -(probs * torch.log(probs + 1e-10)).sum()
        
        # For order >= 2: compute stationary distribution over (order-1)-length contexts
        num_contexts = self.num_symbols ** (self.order - 1)
        
        # Build transition matrix: T[i, j] = P(j | context_i)
        # where j represents "next symbol" and also encodes the new context
        T = self._build_transition_matrix()
        
        # Find stationary distribution via power iteration
        pi = self._compute_stationary_distribution(T, num_contexts)
        
        # Compute entropy: H = -Σ_context π(context) * Σ_symbol P(symbol|context) * log P(symbol|context)
        cond_probs = self.probs[str(self.order - 1)]  # shape: (num_contexts, num_symbols + 1)
        
        # Conditional entropy for each context
        cond_entropy = -(cond_probs * torch.log(cond_probs + 1e-10)).sum(1)  # shape: (num_contexts,)
        
        # Weight by stationary probability of each context
        return (pi * cond_entropy).sum()
    
    def optimize(
        self,
        H_t: float,
        do_logging: bool = True,
        tol: float = 1e-6,
        lr: float = 0.01,
        log_freq: int = 1000,
        max_iter: int = 100_000,
        max_time: float = 300.0,
        K: int = 1000,
        max_retries: int = 4
    ) -> bool:
    
        losses = []
        DH = torch.tensor(H_t, dtype=torch.float32, requires_grad=False, device=self.device)
        criterion = nn.MSELoss()
        if do_logging:
            print(f'criterion: {criterion.__class__.__name__}')
            print(f'Testing {K} random initializations...')

        with torch.no_grad():
            # Normalize current
            probs_norm = {}
            for i in range(self.order):
                probs_norm[str(i)] = self.probs[str(i)].softmax(-1)
            
            probs_orig = {str(i): self.probs[str(i)].clone() for i in range(self.order)}
            
            self.probs = nn.ParameterDict({str(i): nn.Parameter(probs_norm[str(i)]) for i in range(self.order)})
            best_loss = criterion(self.entropy(), DH).item()
            best_probs = {str(i): probs_orig[str(i)].clone() for i in range(self.order)}
            
            # Try K random initializations
            for k in range(K):
                candidate_probs = {}
                for i in range(self.order):
                    candidate_probs[str(i)] = torch.randn(self.probs[str(i)].shape, device=self.device, generator=self.generator)
                
                probs_norm = {str(i): candidate_probs[str(i)].softmax(-1) for i in range(self.order)}
                self.probs = nn.ParameterDict({str(i): nn.Parameter(probs_norm[str(i)]) for i in range(self.order)})
                candidate_loss = criterion(self.entropy(), DH).item()
                
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_probs = {str(i): candidate_probs[str(i)].clone() for i in range(self.order)}
                    if do_logging:
                        print(f'New best at initialization {k}: loss = {best_loss:.6f}')

        if do_logging:
            print(f'Best initialization loss: {best_loss:.6f}')
            print('Starting optimization...')
        
        self.probs = nn.ParameterDict({str(i): nn.Parameter(best_probs[str(i)]) for i in range(self.order)})
        best_optimization_loss = float('inf')
        best_optimization_probs = None
        optimizer = torch.optim.AdamW(self.probs.parameters(), lr=lr)

        i = 0
        start = time()
        while True:
            optimizer.zero_grad()
            
            probs_backup = {str(i): self.probs[str(i)].data.clone() for i in range(self.order)}
            
            for i_order in range(self.order):
                self.probs[str(i_order)].data = self.probs[str(i_order)].softmax(-1)
            
            loss = criterion(self.entropy(), DH)
            loss.backward()
            
            for i_order in range(self.order):
                self.probs[str(i_order)].data = probs_backup[str(i_order)]
            optimizer.step()
            
            if (i % log_freq == 0):
                with torch.no_grad():
                    loss_val = loss.item()
                    if do_logging:
                        print(f'loss: {loss_val:.4}')
                    if loss_val < best_optimization_loss:
                        best_optimization_loss = loss_val
                        best_optimization_probs = {str(j): self.probs[str(j)].clone().detach() for j in range(self.order)}
                    
                    if loss_val < tol:
                        break
                    
                    losses.append(loss_val)
                    if len(losses) > 1 and abs(losses[-1] - losses[-2]) < tol:
                        if losses[-1] >= tol:
                            Warning('Optimization did not converge!')
                        break
                
                if ((time() - start) > max_time) or (i > max_iter):
                    Warning('Optimization did not converge!')
                    break
            
            i += 1

        if best_optimization_probs is not None:
            self.probs = nn.ParameterDict({str(i): nn.Parameter(best_optimization_probs[str(i)]) for i in range(self.order)})

        for i_order in range(self.order):
            self.probs[str(i_order)].data = self.probs[str(i_order)].softmax(-1)

        return True

    def _build_transition_matrix(self) -> torch.Tensor: # GENERATED WITH CLAUDE
        """
        Build a transition matrix where entry [i, j] represents the probability
        of transitioning from context i to context j.
        
        A transition from context i to context j occurs when we:
        - Start in context i (representing symbols s_1, ..., s_{n-1})
        - Generate symbol s_n
        - New context j represents (s_2, ..., s_n)
        """
        num_contexts = self.num_symbols ** (self.order - 1)
        num_symbols = self.num_symbols + 1  # include stop symbol
        
        # T[i, j] = probability of going from context i to context j
        T = torch.zeros(num_contexts, num_contexts, device=self.device)
        
        cond_probs = self.probs[str(self.order - 1)]  # shape: (num_contexts, num_symbols + 1)
        
        for context_idx in range(num_contexts):
            # For each symbol we could generate
            for symbol_idx in range(num_symbols):
                prob = cond_probs[context_idx, symbol_idx]
                
                if symbol_idx == self.num_symbols:
                    # Stop symbol: stay in dummy context (context 0) or distribute over initial states
                    # We treat this as transitioning to a special absorbing state
                    # For entropy purposes, we can ignore paths that terminate
                    continue
                else:
                    # New context: shift context left and add new symbol
                    # context_idx encodes (s_1, ..., s_{n-1})
                    # new context encodes (s_2, ..., s_n) where s_n = symbol_idx
                    new_context_idx = self._context_to_idx(
                        self._idx_to_context(context_idx)[1:] + [symbol_idx]
                    )
                    T[context_idx, new_context_idx] += prob
        
        return T

    def _compute_stationary_distribution(
        self,
        T: torch.Tensor,
        num_contexts: int,
        max_iters: int = 1000,
        tol: float = 1e-6
    ) -> torch.Tensor: # GENERATED WITH CLAUDE
        """
        Compute stationary distribution π such that π = π · T via power iteration.
        
        Initialize π uniformly and iterate until convergence.
        """
        pi = torch.ones(num_contexts, device=self.device) / num_contexts
        
        for _ in range(max_iters):
            pi_new = pi @ T
            
            # Renormalize (may not sum to 1 due to stop symbols)
            pi_new = pi_new / (pi_new.sum() + 1e-10)
            
            if torch.allclose(pi, pi_new, atol=tol):
                break
            
            pi = pi_new
        
        return pi

    def _context_to_idx_helper(self, context: list[int]) -> int:
        if len(context) == 0:
            return 0
        elif len(context) == 1:
            return context[0]
        else:
            return self._context_to_idx_helper(context[:-1]) * self.num_symbols + context[-1]

    def _context_to_idx(self, context: list[int]) -> int:
        while len(context) > 1 and context[0] == 0:
            context = context[1:]
        return self._context_to_idx_helper(context)
    
    def _idx_to_context_helper(self, idx: int) -> list[int]:
        if idx < self.num_symbols:
            return [idx]
        else:
            return self._idx_to_context_helper(idx // self.num_symbols) + [idx % self.num_symbols]
        
    def _idx_to_context(self, idx: int) -> list[int]:
        out = self._idx_to_context_helper(idx)
        while len(out) < (self.order - 1):
            out = [0] + out
        return out


class NGramDataset(SequenceDataset):
    def __init__(
        self,
        grammar: NGram,
        num_seqs: int = 100,
        max_length: int = 128
    ):
        super().__init__(
            grammar=grammar,
            num_seqs=num_seqs,
            max_length=max_length
        )

    
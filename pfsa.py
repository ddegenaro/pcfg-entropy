from time import time
from typing import Union

import torch
from torch import nn

from utils import Sequence, SequenceDataset, Grammar

class PFSA(Grammar):

    def __init__(
        self,
        seed: int = 42,
        device: Union[str, torch.device] = 'cpu',
        chr_ord_offset: int = 97,
        from_file: str = None,
        num_symbols: int = 10,
        num_states: int = 10
    ):
        
        """
        Define a probabilistic finite-state automaton with optimizable probabilities.

        Args:
            `seed`: `int` - random seed to generate reproducible initializations/optimization processes.
            `device`: `str` - PyTorch-compatible device name to run computations on GPUs when possible.
            `chr_ord_offset`: `int` - constant shift to Unicode representations of integer-valued terminal symbols.
            `from_file`: `str` - filepath specifying weights for this PFSA. `num_symbols` and `num_states` will be ignored.
            `num_symbols`: `int` - the number of symbols in the grammar. Used if `from_file` is None.
            `num_states`: `int` - the number of hidden states in the grammar. Used if `from_file` is None.
        """

        # formalism-specific init
        self.formalism = 'pfsa'
        self.num_states: int = num_states

        # load weights and validate
        super().__init__(
            seed,
            device,
            chr_ord_offset,
            from_file,
            num_symbols
        )
        self.file_name_convention = f'pfsa_seed_{self.seed}_symbols_{self.num_symbols}_states_{self.num_states}'

        # formalism-specific data to keep track of, computed last
        self.Q_ordered: list[int] = [
            x for x in range(self.num_states)
        ]
        self.Q = set(self.Q_ordered)

        self.validate()

    def __repr__(self):
        return f'PFSA(seed={self.seed}, num_symbols={self.num_symbols}, num_states={self.num_states})'

    def validate(self):
        super().validate()
        assert self.transitions.shape[0] == self.transitions.shape[2] - 1
        assert self.pi.shape[0] == self.transitions.shape[0]
        assert len(self.Q) == len(self.Q_ordered) == self.num_states
        assert torch.allclose(self.pi.sum(), torch.tensor(1., device=self.device))
        assert torch.allclose(self.transitions.sum(1).sum(1), torch.tensor(1., device=self.device))

    def init_weights(self):

        # self.pi[ord('a') - self.chr_ord_offset] = p(start with 'a')
        self.pi = nn.Parameter(torch.randn(
            (self.num_states,),
            device=self.device,
            generator=self.generator
        ).softmax(0))

        # self.transitions[ord('a') - self.chr_ord_offset, i, j] = p(transition from state i to state j by emitting 'a')
        self.transitions = nn.Parameter(torch.randn(
            (self.num_states, self.num_symbols, self.num_states + 1), # +1 for accept state
            device=self.device,
            generator=self.generator
        ).flatten(start_dim=1).softmax(1).reshape(self.num_states, self.num_symbols, self.num_states + 1))

        super().init_weights() # turn off gradients

    def load(self, fp: str):
        data = torch.load(fp, map_location=self.device)
        self.pi: torch.Tensor = data['pi']
        self.transitions: torch.Tensor = data['transitions']

        # infer num_symbols and num_states from loaded data
        self.num_symbols: int = self.transitions.shape[0]
        self.num_states: int = self.transitions.shape[1]

    def save(self, fp: str):
        torch.save(
            {
                'pi': self.pi.cpu(),
                'transitions': self.transitions.cpu()
            },
            fp
        )
        print(f'Saved pi and transitions to {fp} successfully.')

    def p_seq(self, seq: Union[str, Sequence, list[int]]): # TODO

        tokens = self.tokenize(seq)
        
        return 0
    
    def _generate_one(self, max_length: int) -> Sequence:

        seq = []
        
        curr_state = torch.multinomial(
            self.pi,
            num_samples=1,
            generator=self.generator
        ).item()

        while len(seq) < max_length:
            if curr_state == self.num_states:
                break
            next_symbol, next_state = self._idx_to_symbol_state_pair(
                torch.multinomial(
                    self.transitions[curr_state].flatten(),
                    num_samples=1,
                    generator=self.generator
                ).item()
            )
            seq.append(next_symbol)
            curr_state = next_state
        
        return Sequence(self.untokenize(seq, False))
    
    def entropy(self) -> torch.Tensor: # GENERATED WITH CLAUDE
        """
        Compute entropy over sequences for a PFSA.
        
        The PFSA defines a Markov chain over states. We compute:
        1. Stationary distribution π over non-absorbing states
        2. For each state, the conditional entropy of emissions (symbols + stop)
        3. Total entropy: H = Σ_i π(i) * H(emissions | state i)
        """
        
        # Build state transition matrix: P[i, j] = prob of going from state i to state j
        # (ignoring which symbol was emitted, just the destination state)
        # Shape: (num_states, num_states + 1) where last column is absorbing state
        
        # self.transitions shape: (num_states, num_symbols, num_states + 1)
        # Sum over symbols to get state-to-state transitions
        P = self.transitions.sum(dim=1)  # shape: (num_states, num_states + 1)
        
        # Extract only transitions to non-absorbing states for stationary distribution
        P_non_absorbing = P[:, :-1]  # shape: (num_states, num_states)
        
        # Normalize each row to be a probability distribution over next states
        # (we'll condition on not stopping)
        P_non_absorbing_normalized = P_non_absorbing / (P_non_absorbing.sum(dim=1, keepdim=True) + 1e-10)
        
        # Compute stationary distribution via power iteration
        pi = self._compute_stationary_distribution_pfsa(P_non_absorbing_normalized)
        
        # Compute conditional entropy for each state
        # Each state has a transition matrix of shape (num_symbols, num_states + 1)
        # H(state i) = -Σ_k P(k | state i) * log P(k | state i)
        # where k ranges over all (symbol, next_state) pairs
        
        # Flatten transitions for each state and compute entropy
        # self.transitions[i] has shape (num_symbols, num_states + 1)
        # Flatten to (num_symbols * (num_states + 1),)
        
        conditional_entropies = torch.zeros(self.num_states, device=self.device)
        
        for i in range(self.num_states):
            state_transitions = self.transitions[i]  # shape: (num_symbols, num_states + 1)
            flattened = state_transitions.flatten()  # shape: (num_symbols * (num_states + 1),)
            
            # Compute entropy of this flattened distribution
            conditional_entropies[i] = -(flattened * torch.log(flattened + 1e-10)).sum()
        
        # Total entropy: weight each state's conditional entropy by its stationary probability
        total_entropy = (pi * conditional_entropies).sum()
        
        return total_entropy

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
            pi_norm = self.pi.softmax(0)
            trans_norm = self.transitions.flatten(start_dim=1).softmax(1).reshape(
                self.num_states, self.num_symbols, self.num_states + 1
            )
            
            pi_orig = self.pi.clone()
            trans_orig = self.transitions.clone()
            
            self.pi = nn.Parameter(pi_norm)
            self.transitions = nn.Parameter(trans_norm)
            best_loss = criterion(self.entropy(), DH).item()
            best_pi = pi_orig.clone()
            best_transitions = trans_orig.clone()
            
            # Try K random initializations
            for k in range(K):
                candidate_pi = torch.randn(self.pi.shape, device=self.device, generator=self.generator)
                candidate_trans = torch.randn(self.transitions.shape, device=self.device, generator=self.generator)
                
                pi_norm = candidate_pi.softmax(0)
                trans_norm = candidate_trans.flatten(start_dim=1).softmax(1).reshape(
                    self.num_states, self.num_symbols, self.num_states + 1)
                
                self.pi = nn.Parameter(pi_norm)
                self.transitions = nn.Parameter(trans_norm)
                candidate_loss = criterion(self.entropy(), DH).item()
                
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_pi = candidate_pi.clone()
                    best_transitions = candidate_trans.clone()
                    if do_logging:
                        print(f'New best at initialization {k}: loss = {best_loss:.6f}')

        if do_logging:
            print(f'Best initialization loss: {best_loss:.6f}')
            print('Starting optimization...')
        
        self.pi = nn.Parameter(best_pi)
        self.transitions = nn.Parameter(best_transitions)
        best_optimization_loss = float('inf')
        best_optimization_pi = None
        best_optimization_transitions = None
        optimizer = torch.optim.AdamW([self.pi, self.transitions], lr=lr)

        i = 0
        start = time()
        while True:
            optimizer.zero_grad()
            
            pi_backup = self.pi.data.clone()
            trans_backup = self.transitions.data.clone()
            
            self.pi.data = self.pi.softmax(0)
            self.transitions.data = self.transitions.flatten(start_dim=1).softmax(1).reshape(
                self.num_states, self.num_symbols, self.num_states + 1)
            
            loss = criterion(self.entropy(), DH)
            loss.backward()
            
            self.pi.data = pi_backup
            self.transitions.data = trans_backup
            optimizer.step()
            
            if (i % log_freq == 0):
                with torch.no_grad():
                    loss_val = loss.item()
                    if do_logging:
                        print(f'loss: {loss_val:.4}')
                    if loss_val < best_optimization_loss:
                        best_optimization_loss = loss_val
                        best_optimization_pi = self.pi.clone().detach()
                        best_optimization_transitions = self.transitions.clone().detach()
                    
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

        if best_optimization_pi is not None:
            self.pi = nn.Parameter(best_optimization_pi)
            self.transitions = nn.Parameter(best_optimization_transitions)

        self.pi.data = self.pi.softmax(0)
        self.transitions.data = self.transitions.flatten(start_dim=1).softmax(1).reshape(
            self.num_states, self.num_symbols, self.num_states + 1
        )

        for param in self.parameters():
            param.requires_grad = False

        return True
    
    def _compute_stationary_distribution_pfsa(
        self, P: torch.Tensor, max_iters: int = 1000, tol: float = 1e-8
    ) -> torch.Tensor: # GENERATED WITH CLAUDE
        """
        Compute stationary distribution for PFSA state transitions.
        
        Args:
            P: transition matrix of shape (num_states, num_states) with rows summing to 1
            max_iters: maximum number of power iterations
            tol: convergence tolerance
        
        Returns:
            Stationary distribution π where π = π · P
        """
        
        num_states = P.shape[0]
        
        # Initialize uniformly
        pi = torch.ones(num_states, device=self.device) / num_states
        
        for _ in range(max_iters):
            pi_new = pi @ P
            
            # Renormalize (in case of numerical issues)
            pi_new = pi_new / (pi_new.sum() + 1e-10)
            
            if torch.allclose(pi, pi_new, atol=tol):
                break
            
            pi = pi_new
        
        return pi

    def _idx_to_symbol_state_pair(self, idx: int) -> tuple[int]:
        div = self.num_states + 1
        return (idx // div, idx % div)
    
    def _symbol_state_pair_to_idx(self, symbol_state_pair: tuple[int]) -> int:
        mult = self.num_states + 1
        return symbol_state_pair[0] * mult + symbol_state_pair[1]

class PFSADataset(SequenceDataset):

    def __init__(
        self,
        grammar: PFSA,
        num_seqs: int = 100,
        max_length: int = 128
    ):
        
        super().__init__(
            grammar=grammar,
            num_seqs=num_seqs,
            max_length=max_length
        )

from time import time
from typing import Union

import torch
from torch import nn

from utils import SequenceDataset, Grammar, Node, Tree, Sequence

class PCFG(Grammar):

    def __init__(
        self,
        seed: int = 42,
        device: Union[str, torch.device] = 'cpu',
        chr_ord_offset: int = 97,
        from_file: str = None,
        num_symbols: int = 10,
        num_non_terminals: int = 10,
        do_logging: bool = False
    ):
        
        """
        Define a dense probabilistic context-free grammar in Chomsky Normal Form with optimizable
        probabilities.

        Args:
            `seed`: `int` - random seed to generate reproducible initializations/optimization processes.
            `device`: `str` - PyTorch-compatible device name to run computations on GPUs when possible.
            `chr_ord_offset`: `int` - constant shift to Unicode representations of integer-valued terminal symbols.
            `from_file`: `str` - filepath specifying weights for this PFSA. `num_symbols` and `num_non_terminals` will be ignored.
            `num_symbols`: `int` - the number of terminal symbols (leaf node data) in the grammar.
            `num_non_terminals`: `int` - the number of non-terminal symbols (tree-internal node data) in the grammar.
        """
        
        # formalism-specific inits
        self.formalism = 'pcfg'
        self.num_non_terminals: int = num_non_terminals
        self.S: int = 0

        # load weights and validate
        super().__init__(
            seed,
            device,
            chr_ord_offset,
            from_file,
            num_symbols
        )

        # non-terminals are integers, terminals are utf-8 characters
        self.N_ordered: list[int] = [
            x for x in range(self.S + 1, self.S + self.num_non_terminals + 1)
        ]
        self.N = set(self.N_ordered)
        self.NUS = self.N.union(set([self.S]))
        
        self.file_name_convention = f'pcfg_seed_{self.seed}_symbols_{self.num_symbols}_nts_{self.num_non_terminals}'

        self.validate()

    def __repr__(self):
        return f'PCFG(seed={self.seed}, num_symbols={self.num_symbols}, num_non_terminals={self.num_non_terminals})'

    def validate(self):
        super().validate()
        assert len(self.N) == len(self.N_ordered) == self.num_non_terminals == len(self.NUS) - 1
        if not torch.allclose(self.rules.sum(1), torch.tensor(1., device=self.device)):
            print(f'Warning: probability distributions not summing to 1.')

    def init_weights(self):
        # row and column indices indicate probability of that rule
        # + 1 to num_non_terminals due to start symbol
        self.rules = nn.Parameter(torch.randn(
            (self.num_non_terminals + 1, self.num_symbols + self.num_non_terminals ** 2),
            device=self.device,
            generator=self.generator
        ).softmax(1))

        super().init_weights() # turn off gradients

    def load(self, fp: str):
        self.rules = nn.Parameter(torch.load(fp, map_location=self.device))

        # re-assign and validate
        self.num_symbols: int = self.rules.shape[1] - (self.rules.shape[0] - 1) ** 2
        self.num_non_terminals: int = self.rules.shape[0] - 1
    
    def save(self, fp: str):
        torch.save(self.rules.cpu(), fp)
        print(f'Saved rules to {fp} successfully.')

    def p_seq(self, seq: Union[Sequence, Tree]) -> torch.Tensor:
        val = torch.tensor(1., device=self.device)

        if type(seq) == Sequence:
            tree = seq.data
            assert type(tree) == Tree, f'Cannot compute probability of a non-tree under a PCFG.'
        else:
            tree = seq

        for node in tree.nodes:
            if node.left is None:
                assert node.right is None # no children
            elif node.right is not None:
                val *= self._p_rule(node.data, node.left.data, node.right.data) # L and R
            else:
                val *= self._p_rule(node.data, node.left.data) # just L
        return val
    
    def _generate_one(self, max_length: int) -> Sequence:

        while True:
            tree = Tree(self.S, self.Sigma) # init a tree with start symbol as root
            while self._expand_tree(tree, max_length):
                continue
            # At this point, frontier should be empty and all leaves should be terminals
            if len(tree.leaves) <= max_length:
                assert self._is_expanded(tree)  # This should always pass now
                return Sequence(tree)
            
    def entropy(self) -> torch.Tensor:
        return (
            torch.inverse(torch.eye(self.num_non_terminals + 1, device=self.device) - self._char_matrix())
            @
            (self._local_expansion_vector())
        )[self.S]
    
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
        """
        Optimize parameters to match target entropy H_t.
        
        Args:
            H_t: target entropy
            do_logging: whether to print progress
            tol: convergence tolerance
            lr: learning rate
            log_freq: logging frequency
            max_iter: maximum iterations
            max_time: maximum time in seconds
            K: number of random initializations to try
        
        Returns:
            True if optimization succeeded
        """
        if do_logging:
            print('-----------------------------------------------------')

        losses = []
        
        DH = torch.tensor(H_t, dtype=torch.float32, requires_grad=False, device=self.device)
        criterion = nn.MSELoss()
        if do_logging:
            print(f'criterion: {criterion.__class__.__name__}')
            print(f'Testing {K} random initializations...')

        with torch.no_grad():
            # Compute initial loss with current rules
            normalized_current = self.rules.softmax(1)
            original_rules = self.rules
            self.rules = nn.Parameter(normalized_current)
            best_loss = criterion(self.entropy(), DH).item()
            best_rules = original_rules.clone()  # Store the raw rules
            
            # Try K random initializations
            for k in range(K):
                # Generate random tensor of same shape (raw values)
                candidate_rules = torch.randn(
                    self.rules.shape,
                    device=self.device,
                    generator=self.generator
                )
                candidate_normalized = candidate_rules.softmax(1)
                
                # Compute loss with candidate rules
                self.rules = nn.Parameter(candidate_normalized)
                candidate_loss = criterion(self.entropy(), DH).item()
                
                # Update best if this is better
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_rules = candidate_rules.clone()  # Store raw values
                    if do_logging:
                        print(f'New best at initialization {k}: loss = {best_loss:.6f}')

        if do_logging:
            print(f'Best initialization loss: {best_loss:.6f}')
            print('Starting optimization...')
        
        self.rules = nn.Parameter(best_rules)
        best_optimization_loss = float('inf')
        best_optimization_rules = None
        optimizer = torch.optim.AdamW([self.rules], lr=lr)

        i = 0
        start = time()
        while True:
            optimizer.zero_grad()
            
            # Temporarily apply softmax for entropy computation
            rules_backup = self.rules.data.clone()
            self.rules.data = self.rules.softmax(1)
            
            loss = criterion(self.entropy(), DH)
            loss.backward()
            
            # Restore raw before step
            self.rules.data = rules_backup
            optimizer.step()
            
            if (i % log_freq == 0):
                with torch.no_grad():

                    loss_val = loss.item()
                    
                    if do_logging:
                        print(f'loss: {loss_val:.4}', end='\r')
                    if loss_val < best_optimization_loss:
                        best_optimization_loss = loss_val
                        best_optimization_rules = self.rules.clone().detach()

                    if loss_val < tol:
                        break
                    
                    losses.append(loss_val)
                    if len(losses) > 1:
                        if abs(losses[-1] - losses[-2]) < tol:
                            if losses[-1] >= tol:
                                msg = 'Optimization did not converge!'
                                Warning(msg)
                            break
                        
                if ((time() - start) > max_time) or (i > max_iter):
                    msg = 'Optimization did not converge!'
                    Warning(msg)
                    break
            
            i += 1

        # Restore best at the end
        if best_optimization_rules is not None:
            self.rules = nn.Parameter(best_optimization_rules)

        self.rules.data = self.rules.data.softmax(1)

        rho = self._char_matrix_rho()
        print(f'Spectral radius: {rho}.', end=' ')

        if rho < 1.0:
            print('Done.')
            for param in self.parameters():
                param.requires_grad = False
            return True
        elif max_retries == 0:
            print('No more retries.')
            for param in self.parameters():
                param.requires_grad = False
            return False
        else:
            print('Retrying...')
            return self.optimize(
                H_t,
                do_logging,
                tol,
                lr,
                log_freq,
                max_iter,
                max_time,
                K,
                max_retries - 1
            )
    
    def _symbols_to_column_index(self, symbol1: Union[str, int], symbol2: int = None) -> int:

        """
        Return the integer index from the rules given a string representation of a terminal symbol,
        or a pair of integer non-terminal symbols.
        """

        if symbol2 is None:
            # assert symbol1 in self.Sigma, f"{symbol1} is not a terminal."
            return ord(symbol1) - self.chr_ord_offset
        else:
            # assert symbol1 in self.N, f"{symbol1} is not a non-terminal (or is the start symbol)."
            # assert symbol2 in self.N, f"{symbol2} is not a non-terminal (or is the start symbol)."
            # - 1 accounts for excluding start
            return self.num_symbols + (symbol1 - 1) * self.num_non_terminals + (symbol2 - 1)

    def _column_index_to_symbols(self, index: int) -> Union[str, tuple[int]]:
        """
        Return the string representation of a terminal symbol, or a pair of integer non-terminal symbols,
        given an integer index from the rules.
        """
        
        if self._col_is_terminal(index):
            return chr(index + self.chr_ord_offset)
        else:
            offset = index - self.num_symbols # adding 1 solves excluding start symbol
            return (1 + (offset // self.num_non_terminals), 1 + (offset % self.num_non_terminals))
        
    def _col_is_terminal(self, index: int) -> bool:

        """
        Check whether this index from the rules represents a terminal rule.
        """

        # assert index >= 0, f'Index {index} should not be negative'

        return index < self.num_symbols
    
    def _p_rule(
        self, 
        non_terminal: int,
        symbol1: Union[str, int] = None,
        symbol2: int = None
    ) -> torch.Tensor:

        """
        Get the probability of a non-terminal symbol being rewritten as the terminal symbol or pair of
        non-terminal symbols provided.
        """                
        
        # assert non_terminal in self.NUS, f"{non_terminal} is not a non-terminal."
        if symbol2 is None:
            return self.rules[non_terminal, self._symbols_to_column_index(symbol1)]
        else:
            return self.rules[non_terminal, self._symbols_to_column_index(symbol1, symbol2)]

    def _expand_tree(self, tree: Tree, max_length: int) -> bool:
        if len(tree.frontier) > 0:
            expansion_point: Node = tree.frontier.pop(0)
            symbol = expansion_point.data
            # assert symbol in self.NUS, f"Found terminal symbol {symbol} in frontier."
            sampled_rule_index: int = torch.multinomial(
                self.rules[symbol], num_samples = 1, generator=self.generator
            ).item()
            symbol_s = self._column_index_to_symbols(sampled_rule_index)
            if self._col_is_terminal(sampled_rule_index):
                tree.add_left( # add a single node with the expansion point as parent
                    Node(symbol_s),
                    expansion_point
                )
                if len(tree.leaves) > max_length:
                    return False
            else:
                nt1, nt2 = symbol_s
                tree.add_pair( # add a pair of nodes with the expansion point as parent
                    Node(nt1),
                    Node(nt2),
                    expansion_point
                )
                tree.frontier.extend([expansion_point.left, expansion_point.right]) # expand frontier
                if len(tree.leaves) > max_length:
                    return False
            return True
        return False

    def _is_expanded(self, tree: Tree) -> bool:
        # for leaf in tree.leaves:
        #     if leaf.data not in self.Sigma:
        #         tree.show()
        #         raise Exception(f'Found a non-terminal leaf: {leaf.data} which is not in {self.Sigma}')
        return len(tree.frontier) == 0

    def _char_matrix(self) -> torch.Tensor:
        # Extract binary rules and reshape to (|NUS|, |N|, |N|)
        binary_rules = self.rules[:, self.num_symbols:].view(
            self.num_non_terminals + 1,
            self.num_non_terminals,
            self.num_non_terminals
        )
        
        # Sum over dimensions to count occurrences of each non-terminal
        m = torch.zeros(
            (self.num_non_terminals + 1, self.num_non_terminals + 1),
            device=self.device
        )

        m[:, 1:] = binary_rules.sum(dim=2) + binary_rules.sum(dim=1)
        
        return m

    def _local_expansion_vector(self) -> torch.Tensor:
        return -(self.rules * self.rules.log()).sum(1)
    
    def _length_expansion_vector(self) -> torch.Tensor:
        pass # TODO
    
    def _char_matrix_rho(self) -> torch.Tensor:

        # eigenvalues must be done on CPU, not implemented on MPS
        return torch.linalg.eigvals(self._char_matrix().cpu()).abs().max()

    def cky(self, v: Union[str, list[int], torch.Tensor, Tree]) -> torch.Tensor:

        if type(v) in (str, Tree):
            w = self.tokenize(v)
        else:
            w = v

        # inside probabilities
        B = torch.zeros(
            (len(w), len(w), self.rules.shape[0]),
            device = self.device
        )
        B[range(len(w)), range(len(w)), :] = self.rules[:, w].T

        # precompile indices
        YZ_indices = torch.arange(self.num_symbols, self.rules.shape[1], device = self.device)
        YZ_pairs = [
            self._column_index_to_symbols(YZ) for YZ in range(self.num_symbols, self.rules.shape[1])
        ]
        Y_vals, Z_vals = torch.tensor(YZ_pairs, device = self.device).T

        for l in range(2, len(w)+1):
            for i in range(0, len(w) - l + 1):
                k = i + l - 1
                j_range = torch.arange(i, k)
                
                # Vectorized over all YZ at once
                S = torch.sum(
                    B[i, j_range, Y_vals[:, None]] * B[j_range+1, k, Z_vals[:, None]],
                    dim=1
                )
                
                # Vectorized update
                B[i, k, :] += torch.sum(self.rules[:, YZ_indices] * S[None, :], dim=1)

        # assert B[0, len(w)-1, 0] > 0, f'Got probability {B[0, len(w)-1, 0].item()} for the given string.'

        return B

    def _P(self) -> torch.Tensor:
        P = torch.zeros((self.num_non_terminals + 1, self.num_non_terminals + 1))

        # Reshape the non-terminal part of rules matrix
        non_terminal_rules = self.rules[:, self.num_symbols:].view(
            self.num_non_terminals + 1,
            self.num_non_terminals,
            self.num_non_terminals
        )

        # Sum over the last dimension (all possible second non-terminals)
        P[:, 1:] = non_terminal_rules.sum(dim=2)

        return P

    def _E_lc_one_symbol(self) -> torch.Tensor:
        return torch.inverse(
            torch.eye(self.num_non_terminals + 1, device=self.device) - self._P()
        )



class PCFGDataset(SequenceDataset):

    def __init__(
        self,
        grammar: PCFG,
        num_seqs: int = 100,
        max_length: int = 128,
        do_logging: bool = False
    ):
        
        super().__init__(
            grammar=grammar,
            num_seqs=num_seqs,
            max_length=max_length,
            do_logging=do_logging
        )
from time import time
from typing import Union, Iterable
from concurrent.futures import ThreadPoolExecutor

import torch
import networkx as nx
import matplotlib.pyplot as plt

from utils import SequenceDataset

class Node:

    def __init__(self, data):
        self.data = data
        self.parent = None
        self.left = None
        self.right = None



class Tree:

    def __init__(self, root_data, grammar):
        self.root = Node(root_data)
        self.frontier = [self.root]
        self.nodes = set([self.root])
        self.edges = set()
        self.leaves = set([self.root])
        self.grammar = grammar

    def __len__(self):
        return len(self.leaves)
    
    def __str__(self):
        return self.grammar.flatten_to_str(self)

    def add_left(self, n: Node, parent: Node):
        self.nodes.add(n)
        self.edges.add((parent, n)) # from parent to child
        parent.left = n # only child
        n.parent = parent
        self.leaves.remove(parent)
        self.leaves.add(n)

    def add_pair(self, n1: Node, n2: Node, parent: Node):
        self.nodes.add(n1)
        self.nodes.add(n2)
        self.edges.add((parent, n1)) # from parent to child
        self.edges.add((parent, n2)) # from parent to child
        parent.left = n1 # left child
        parent.right = n2 # right child
        n1.parent = parent
        n2.parent = parent
        self.leaves.remove(parent)
        self.leaves.add(n1)
        self.leaves.add(n2)
    
    def tree_positions(self, G, root):
        """
        Simple function to position nodes in a binary tree for clean visualization.
        
        Args:
            G: NetworkX graph representing the tree
            root: Root node of the tree
        
        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        pos = {}
        
        def position_node(node, x, y, width, parent=None):
            # Position current node
            pos[node] = (x, y)
            
            # Get left and right children specifically
            left_child = getattr(node, 'left', None)
            right_child = getattr(node, 'right', None)
            
            if left_child and right_child:
                # Both children: left goes left, right goes right
                left_x = x - width / 2
                right_x = x + width / 2
                position_node(left_child, left_x, y - 1, width / 2, node)
                position_node(right_child, right_x, y - 1, width / 2, node)
            elif left_child:
                # Only left child: place it directly below parent
                position_node(left_child, x, y - 1, width / 2, node)
            elif right_child:
                # Only right child: place it on the right
                right_x = x + width / 4
                position_node(right_child, right_x, y - 1, width / 2, node)
        
        # Start positioning from root
        position_node(root, 0, 0, 8)  # Start with larger initial width
        
        return pos

    def show(self):
        G = nx.Graph()
        G.add_nodes_from(sorted(list(self.nodes), key = lambda x: ord(x.data) if type(x.data) == str else x.data))
        G.add_edges_from(self.edges)
        label_dict = {}
        for node in self.nodes:
            if type(node.data) == str:
                label_dict[node] = node.data
            else:
                label_dict[node] = 'N' + str(node.data)

        plt.figure(figsize=(24, 12))
        # pos = graphviz_layout(G, prog='twopi', root=self.root)
        pos = self.tree_positions(G, self.root)
        nx.draw(
            G,
            pos,
            labels=label_dict,
            with_labels=True,
            node_color='skyblue',
            edge_color='gray',
            node_size=2000,
            font_size=16
        )
        plt.show()



class PCFG:

    def __init__(
        self,
        num_non_terminals: int = 10,
        num_terminals: int = 10,
        chr_ord_offset: int = 97,
        rules: Union[torch.Tensor, str] = None,
        device: str = None,
        seed: int = 42
    ):
        
        """
        Define a dense probabilistic context-free grammar in Chomsky Normal Form with optimizable
        probabilities.

        Args:
            `num_non_terminals`: `int` - the number of non-terminal symbols in the grammar.
            `num_terminals`: `int` - the number of terminal symbols in the grammar.
            `chr_ord_offset`: `int` - constant shift to Unicode representations of integer-valued terminal symbols.
            `rules`: `torch.Tensor | str` - probabilities for weighting the rules. Generated if not passed.
            `device`: `str` - PyTorch-compatible device name to run computations on GPUs when possible.
            `seed`: `int` - random seed to generate reproducible initializations/optimization processes.
        """

        self.cpu_generator = torch.Generator('cpu')

        if torch.backends.mps.is_available():
            self.gpu_generator = torch.Generator('mps')
        elif torch.cuda.is_available():
            self.gpu_generator = torch.Generator('cuda')
        else:
            self.gpu_generator = None

        if seed is not None:
            self.cpu_generator.manual_seed(seed)
            if self.gpu_generator is not None:
                self.gpu_generator.manual_seed(seed)

        if device is None or device == 'cpu' or device == torch.device('cpu') or self.gpu_generator is None:
            self.generator = self.cpu_generator
        else:
            self.generator = self.gpu_generator

        # start symbol is always 0
        self.start_symbol = 0
        self.chr_ord_offset = chr_ord_offset
        if rules is None:

            # row and column indices indicate probability of that rule
            self.rules = torch.randn(
                (num_non_terminals + 1, num_terminals + num_non_terminals ** 2),
                device=device,
                generator=self.generator
            ).softmax(1)

            self.num_non_terminals: int = num_non_terminals
            self.num_terminals: int = num_terminals

        else:
        
            if type(rules) == str:
                self.rules = torch.nn.Parameter(torch.load(rules, map_location=device))
            elif type(rules) == torch.Tensor:
                self.rules = rules
            else:
                raise TypeError(f"Expected rules to be a string or Torch tensor but got: {type(rules)}")

            self.num_non_terminals: int = self.rules.shape[0] - 1
            self.num_terminals: int = self.rules.shape[1] - self.num_non_terminals ** 2

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        # non-terminals are integers, terminals are utf-8 characters
        self.non_terminals_ordered: list[int] = [
            x for x in range(self.start_symbol + 1, self.start_symbol + self.num_non_terminals + 1)
        ]
        self.terminals_ordered: list[str] = [
            chr(x + chr_ord_offset) for x in range(self.num_terminals)
        ]

        # pre-compile for quick lookup, also uses popular naming conventions
        self.N = set(self.non_terminals_ordered)
        assert len(self.N) == len(self.non_terminals_ordered) == self.num_non_terminals
        self.Sigma = set(self.terminals_ordered)
        assert len(self.Sigma) == len(self.terminals_ordered) == self.num_terminals
        self.S = self.start_symbol
        self.NUS = self.N.union(set([self.S]))
        self.pad_id = self.num_terminals

        if not torch.allclose(self.rules.sum(1), torch.tensor(1., device=self.device)):
            print(f'Warning: probability distributions not summing to 1.')

        rho = self._char_matrix_rho()
        # if rho >= 1.0:
        #     warn(f'The characteristic matrix has spectral radius {rho} ≥ 1.0, which may cause undesirable behavior.')

    def _preorder(self, node: Node | None) -> str:
        if node is None:
            return ''
        
        result = ''
        
        # Check current node first
        if node.data in self.Sigma:
            result += node.data
        
        # Then process left and right subtrees
        result += self._preorder(node.left)
        result += self._preorder(node.right)
        
        return result

    def flatten_to_str(self, tree: Tree) -> str:
        return self._preorder(tree.root)
    
    def tokenize(self, tree: Union[Tree, str], return_tensors=None) -> Union[list[int], torch.Tensor]:

        if type(tree) == Tree:
            ids = [
                self._symbols_to_column_index(s) for s in self.flatten_to_str(tree)
            ]
        else:
            ids = [self._symbols_to_column_index(s) for s in tree]

        if return_tensors == 'pt':
            return torch.tensor(ids, dtype=int, device=self.device)
        else:
            return ids
        
    def batch_tokenize(
        self,
        trees: Iterable[Tree],
        return_tensors=None,
        truncate_length=None
    ) -> dict[str, Union[list[int], torch.Tensor]]:
        
        input_ids = [self.tokenize(t) for t in trees]

        if truncate_length is not None:
            for i in range(len(input_ids)):
                input_ids[i] = input_ids[i][:truncate_length]

        pad_mask = [[1 for _ in id_list] for id_list in input_ids]

        longest = 0
        for id_list in input_ids:
            length = len(id_list)
            if length > longest:
                longest = length
        
        for i in range(len(input_ids)):
            while len(input_ids[i]) < longest:
                input_ids[i].append(self.pad_id)
                pad_mask[i].append(0)

        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=int, device=self.device)
            pad_mask = torch.tensor(pad_mask, dtype=int, device=self.device)

        return {
            'input_ids': input_ids,
            'attention_mask': pad_mask
        }

    def untokenize(self, seq: Union[list[int], torch.Tensor]) -> str:

        if type(seq) == torch.Tensor:
            as_list = seq.tolist()
        else:
            as_list = seq

        return ''.join([self._column_index_to_symbols(s) for s in as_list])

    def _symbols_to_column_index(self, symbol1: Union[str, int], symbol2: int = None) -> int:

        """
        Return the integer index from the rules given a string representation of a terminal symbol,
        or a pair of integer non-terminal symbols.
        """

        if symbol2 is None:
            assert symbol1 in self.Sigma, f"{symbol1} is not a terminal."
            return ord(symbol1) - self.chr_ord_offset
        else:
            assert symbol1 in self.N, f"{symbol1} is not a non-terminal (or is the start symbol)."
            assert symbol2 in self.N, f"{symbol2} is not a non-terminal (or is the start symbol)."
            # - 1 accounts for excluding start
            return self.num_terminals + (symbol1 - 1) * self.num_non_terminals + (symbol2 - 1)

    def _column_index_to_symbols(self, index: int) -> Union[str, tuple[int]]:
        """
        Return the string representation of a terminal symbol, or a pair of integer non-terminal symbols,
        given an integer index from the rules.
        """
        
        if self._col_is_terminal(index):
            return chr(index + self.chr_ord_offset)
        else:
            offset = index - self.num_terminals # adding 1 solves excluding start symbol
            return (1 + (offset // self.num_non_terminals), 1 + (offset % self.num_non_terminals))
        
    def _col_is_terminal(self, index: int) -> bool:

        """
        Check whether this index from the rules represents a terminal rule.
        """

        assert index >= 0, f'Index {index} should not be negative'

        return index < self.num_terminals
    
    def p(
        self, 
        non_terminal: int,
        symbol1: Union[str, int] = None,
        symbol2: int = None
    ) -> torch.Tensor:

        """
        Get the probability of a non-terminal symbol being rewritten as the terminal symbol or pair of
        non-terminal symbols provided.
        """                
        
        assert non_terminal in self.NUS, f"{non_terminal} is not a non-terminal."
        if symbol2 is None:
            return self.rules[non_terminal, self._symbols_to_column_index(symbol1)]
        else:
            return self.rules[non_terminal, self._symbols_to_column_index(symbol1, symbol2)]

    def p_tree(
        self,
        tree: Tree
    ):
        val = torch.tensor(1., device=self.device)
        for node in tree.nodes:
            if node.left is None:
                assert node.right is None # no children
            elif node.right is not None:
                val *= self.p(node.data, node.left.data, node.right.data) # L and R
            else:
                val *= self.p(node.data, node.left.data) # just L
        return val
    
    def _expand_tree(self, tree: Tree, max_length: int) -> bool:
        if len(tree.frontier) > 0:
            expansion_point: Node = tree.frontier.pop(0)
            symbol = expansion_point.data
            assert symbol in self.NUS, f"Found terminal symbol {symbol} in frontier."
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
        for leaf in tree.leaves:
            if leaf.data not in self.Sigma:
                tree.show()
                raise Exception(f'Found a non-terminal leaf: {leaf.data} which is not in {self.Sigma}')
        return len(tree.frontier) == 0

    def _generate_one(self, max_length: int) -> Tree:

        while True:
            tree = Tree(self.S, self) # init a tree with start symbol as root
            while self._expand_tree(tree, max_length):
                continue
            # At this point, frontier should be empty and all leaves should be terminals
            if len(tree.leaves) <= max_length:
                assert self._is_expanded(tree)  # This should always pass now
                return tree

    def generate(
        self,
        max_length: int = 128,
        num_seqs: int = 1,
        max_threads: int = None
    ) -> list[Tree]:
        
        if max_length is None or max_length <= 0:
            max_length = torch.inf

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [
                executor.submit(self._generate_one, max_length)
                for _ in range(num_seqs)
            ]
            trees = [future.result() for future in futures]

        return trees
    
    def _char_matrix(self) -> torch.Tensor:
        # Extract binary rules and reshape to (|NUS|, |N|, |N|)
        binary_rules = self.rules[:, self.num_terminals:].view(
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

    def entropy(self) -> torch.Tensor:
        return (
            torch.inverse(torch.eye(self.num_non_terminals + 1, device=self.device) - self._char_matrix())
            @
            (self._local_expansion_vector())
        )[self.S]
    
    def _char_matrix_rho(self) -> torch.Tensor:

        # eigenvalues must be done on CPU, not implemented on MPS
        return torch.linalg.eigvals(self._char_matrix().cpu()).abs().max()

    def optimize(
        self,
        H_t: float,
        do_logging: bool = True,
        tol: float = 1e-6,
        lr: float = 0.001,
        log_freq: int = 1000,
        max_iter: int = 100_000,
        max_time: float = 300.0,
        K: int = 1000,
        max_retries: int = 4
    ) -> tuple[torch.Tensor, float, float, list[float]]:
        
        losses = []
        
        DH = torch.tensor(H_t, dtype=torch.float32, requires_grad=False, device=self.device)
        criterion = torch.nn.MSELoss()
        if do_logging:
            print(f'criterion: {criterion.__class__.__name__}')
            print(f'Testing {K} random initializations...')

        with torch.no_grad():
            # Compute initial loss with current rules
            normalized_current = self.rules.softmax(1)
            original_rules = self.rules
            self.rules = normalized_current
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
                self.rules = candidate_normalized
                candidate_loss = criterion(self.entropy(), DH).item()
                
                # Update best if this is better
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_rules = candidate_rules.clone()  # Store raw values
                    if do_logging:
                        print(f'New best at initialization {k}: loss = {best_loss:.6f}')
        
        # Set rules to best found initialization
        self.rules = best_rules
        if do_logging:
            print(f'Best initialization loss: {best_loss:.6f}')
            print('Starting optimization...')
        
        self.rules = torch.nn.Parameter(self.rules)
        best_optimization_loss = float('inf')
        best_optimization_rules = None
        optimizer = torch.optim.AdamW([self.rules], lr=lr)
        
        i = 0
        if do_logging:
            print('-----------------------------------------------------')
        start = time()
        while True:
            
            optimizer.zero_grad()
            
            normalized_rules = self.rules.softmax(1)
            original_rules = self.rules
            self.rules = normalized_rules
            loss = criterion(self.entropy(), DH)
            self.rules = original_rules
            loss.backward()
            optimizer.step()
            
            if (i % log_freq == 0):
                with torch.no_grad():

                    loss_val = loss.item()
                    
                    if do_logging:
                        print(f'loss: {loss_val:.4}')
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
        
        with torch.no_grad():
            if best_optimization_rules is not None:
                self.rules = best_optimization_rules.softmax(1).detach()
            else:
                self.rules = self.rules.softmax(1).detach()

        rho = self._char_matrix_rho()
        print(f'Spectral radius: {rho}.')

        if rho < 1.0:
            return True
        elif max_retries == 0:
            return False
        else:
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

    def to(self, device: Union[str, torch.device]):
        self.rules = self.rules.to(device)
        self.device = device
        if device == 'cpu' or device == torch.device('cpu'):
            self.generator = self.cpu_generator
        else:
            self.generator = self.gpu_generator
            if self.generator is None:
                print(f'No GPU generator available - does this machine have a GPU?')
        return self
    
    def save(self, fp: str):
        torch.save(self.rules.cpu(), fp)
        print(f'Saved rules to {fp} successfully.')

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
        YZ_indices = torch.arange(self.num_terminals, self.rules.shape[1], device = self.device)
        YZ_pairs = [
            self._column_index_to_symbols(YZ) for YZ in range(self.num_terminals, self.rules.shape[1])
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
        non_terminal_rules = self.rules[:, self.num_terminals:].view(
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

    def jl(self, v: Union[str, list[int], torch.Tensor]) -> torch.Tensor:

        if type(v) == str:
            w = self.tokenize(v)
        else:
            w = v

        p_pi = torch.zeros(
            len(w),
            len(w),
            self.rules.shape[0]
        )

        B = self.cky(w)

        E_lc_one_symbol = self._E_lc_one_symbol()

        non_terminal_rules = self.rules[:, self.num_terminals:].view(
            self.num_non_terminals + 1,
            self.num_non_terminals,
            self.num_non_terminals
        )

        E_lc_two_symbols = non_terminal_rules.T @ E_lc_one_symbol

        

        raise NotImplementedError

    def fast_jl(self, v: Union[str, list[int], torch.Tensor]) -> torch.Tensor:

        if type(v) == str:
            w = self.tokenize(v)
        else:
            w = v

        p_pi = 0

        raise NotImplementedError



class PCFGDataset(SequenceDataset):

    def __init__(
        self,
        grammar: PCFG,
        num_seqs: int = 100,
        max_length: int = 128
    ):
        
        super().__init__(
            grammar=grammar,
            num_seqs=num_seqs,
            max_length=max_length
        )

    def example_to_str(self, seq):
        return self.grammar.flatten_to_str(seq)
    
    

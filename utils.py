from random import Random
from typing import Union, Iterable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import torch
from torch import nn
import networkx as nx
import matplotlib.pyplot as plt

class Node:

    def __init__(self, data):
        self.data = data
        self.parent: Node = None
        self.left: Node = None
        self.right: Node = None



class Tree:

    def __init__(self, root_data, Sigma: set):
        self.root = Node(root_data)
        self.frontier = [self.root]
        self.nodes = set([self.root])
        self.edges = set()
        self.leaves = set([self.root])
        self.Sigma = Sigma

    def __len__(self):
        return len(self.leaves)

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
    
    def tree_positions(self, root):
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
        pos = self.tree_positions(self.root)
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

    def _preorder(self, node: Union[Node, None]) -> str:
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
    
    def __str__(self):
        return self._preorder(self.root)



class Sequence:
    
    def __init__(self, data: Union[str, Tree]):
        self.data = data
        if type(self.data) == str:
            self.str_data = self.data
        elif type(self.data) == Tree:
            self.str_data = str(self.data)
        else:
            raise TypeError(f'Sequence should be instantiated by a string or a tree, not a {type(data)}')

    def __str__(self):
        return self.str_data
    
    def __repr__(self):
        return f'Sequence({str(self)})'
    
    def __len__(self):
        return len(self.str_data)
    
    def __getitem__(self, idx):
        return self.str_data[idx]



class Grammar(nn.Module):

    def __init__(
        self,
        seed: int = 42,
        device: Union[str, torch.device] = 'cpu',
        chr_ord_offset: int = 97,
        from_file: str = None,
        num_symbols: int = 10
    ):
        
        """
        Superclass for all grammar formalisms.

        Args:
            `seed`: `int` - random seed to generate reproducible initializations/optimization processes.
            `device`: `str` - PyTorch-compatible device name to run computations on GPUs when possible.
            `chr_ord_offset`: `int` - constant shift to Unicode representations of integer-valued terminal symbols.
            `from_file`: `str` - filepath specifying weights for this PFSA. `num_symbols` and `num_non_terminals` will be ignored.
            `num_symbols`: `int` - the number of terminal symbols (leaf node data) in the grammar.
        """

        super().__init__()

        self.seed: int = seed
        self.device: Union[str, torch.device] = device
        self.chr_ord_offset: int = chr_ord_offset
        self.from_file: str = from_file
        self.num_symbols: int = num_symbols # used only if from_file is None

        self.cpu_generator = torch.Generator('cpu')
        if torch.backends.mps.is_available():
            self.gpu_generator = torch.Generator('mps')
        elif torch.cuda.is_available():
            self.gpu_generator = torch.Generator('cuda')
        else:
            self.gpu_generator = None

        if self.seed is not None:
            self.cpu_generator.manual_seed(self.seed)
            if self.gpu_generator is not None:
                self.gpu_generator.manual_seed(self.seed)

        if self.device is None or self.device == 'cpu' or self.device == torch.device('cpu') or self.gpu_generator is None:
            self.generator = self.cpu_generator
        else:
            self.generator = self.gpu_generator

        if from_file is not None:
            self.load(from_file)
        else:
            self.init_weights()

        self.eos_id = self.num_symbols + 1
        self.pad_id = self.num_symbols + 2
        self.symbols_ordered: list[str] = [
            chr(x + chr_ord_offset) for x in range(self.num_symbols)
        ]
        self.Sigma = set(self.symbols_ordered)

    def validate(self):
        assert len(self.Sigma) == len(self.symbols_ordered) == self.num_symbols

    def init_weights(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def load(self, fp: str):
        raise NotImplementedError

    def save(self, fp: str):
        raise NotImplementedError
    
    def p_seq(self, seq: Union[str, Sequence, list[int]]) -> torch.Tensor:
        raise NotImplementedError

    def _generate_one(self, max_length: int) -> Sequence:
        raise NotImplementedError
    
    def entropy(self) -> torch.Tensor:
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError
    
    def estimated_mls(self):
        tol = 1e-3
        last_estimate = -1.
        total_seqs = 0
        total_chars = 0
        while True:
            total_seqs += 1000
            total_chars += sum([len(x) for x in self.generate(num_seqs=1000)])
            new_estimate = total_chars / total_seqs
            diff = abs(new_estimate - last_estimate)
            # print(diff, total_chars, total_seqs)
            if diff < tol:
                break
            last_estimate = new_estimate
        return new_estimate

    def tokenize(
        self,
        seq: Union[str, Sequence, list[int]],
        return_tensors: str = None
    ) -> Union[list[int], torch.Tensor]:
        
        if type(seq) == str:
            ids = [
                ord(x) - self.chr_ord_offset for x in seq
            ]
        elif type(seq) == Sequence:
            ids = [
                ord(x) - self.chr_ord_offset for x in str(seq)
            ]
        elif type(seq) == list:
            ids = seq
        else:
            raise TypeError(f'Cannot tokenize an object of type {type(seq)}')

        if return_tensors == 'pt':
            return torch.tensor(ids, dtype=int, device=self.device)
        else:
            return ids
        
    def batch_tokenize(
        self,
        seqs: Iterable[Sequence],
        return_tensors=None,
        truncate_length=None
    ) -> dict[str, Union[list[int], torch.Tensor]]:
        
        input_ids = [self.tokenize(seq) for seq in seqs]

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
    
    def untokenize(
        self,
        seq: Union[list[int], torch.Tensor],
        return_sequence=True
    ) -> Union[str, Sequence]:

        if type(seq) == torch.Tensor:
            as_list = seq.tolist()
        else:
            as_list = seq

        s_rep = ''.join([chr(symbol + self.chr_ord_offset) for symbol in as_list])

        if return_sequence:
            return Sequence(s_rep)
        else:
            return s_rep

    def to(self, device: Union[str, torch.device]):
        for param in self.parameters():
            param.to(device)
        self.device = device
        if device == 'cpu' or device == torch.device('cpu'):
            self.generator = self.cpu_generator
        else:
            self.generator = self.gpu_generator
            if self.generator is None:
                print(f'No GPU generator available - does this machine have a GPU?')
        return self

    def generate(
        self,
        max_length: int = 128,
        num_seqs: int = 1,
        max_threads: int = None,
        do_logging: bool = False,
        fp: str = None
    ) -> Union[list[Sequence], Generator]:
        
        if max_length is None or max_length <= 0:
            max_length = torch.inf

        if fp is None:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = [
                    executor.submit(self._generate_one, max_length)
                    for _ in range(num_seqs)
                ]
                if do_logging:
                    seqs = [
                        future.result() 
                        for future in tqdm(as_completed(futures), total=num_seqs)
                    ]
                else:
                    seqs = [future.result() for future in futures]

            return seqs
        else:
            with open(fp, 'w+', encoding='utf-8') as f:
                with ThreadPoolExecutor(max_workers=max_threads) as executor:
                    futures = [
                        executor.submit(self._generate_one, max_length)
                        for _ in range(num_seqs)
                    ]
                    
                    if do_logging:
                        iterable = tqdm(as_completed(futures), total=num_seqs)
                    else:
                        iterable = as_completed(futures)
                    
                    for future in iterable:
                        f.write(str(future.result()) + '\n')

            with open(fp, 'r', encoding='utf-8') as f:
                for line in f:
                    yield Sequence(line.strip())

    def _get_param_copy(self) -> dict:
        """
        Create a deep copy of all parameters in the appropriate format.
        Returns a dict mapping parameter names to cloned tensors.
        """
        params_copy = {}
        for name, param in self.named_parameters():
            params_copy[name] = param.data.clone().detach()
        return params_copy

    def _set_params_from_copy(self, params_copy: dict):
        """
        Set parameters from a saved copy dictionary.
        """
        for name, value in params_copy.items():
            # Navigate to the parameter and set it
            module_name, param_name = name.rsplit('.', 1) if '.' in name else ('', name)
            if module_name:
                module = self
                for attr in module_name.split('.'):
                    module = getattr(module, attr)
                setattr(module, param_name, nn.Parameter(value.clone()))
            else:
                setattr(self, param_name, nn.Parameter(value.clone()))



class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        grammar: Grammar,
        num_seqs: int = 100,
        max_length: int = 128,
        do_logging: bool = True,
        fp: str = None
    ):
        
        super().__init__()

        if max_length is None or max_length <= 0:
            max_length = torch.inf
        self.grammar = grammar
        self.max_length = max_length
        self.num_seqs = num_seqs
        self.fp = fp
        
        if self.fp is None:
            self.seq_list = self.grammar.generate(
                max_length=max_length,
                num_seqs=num_seqs,
                do_logging=do_logging
            )
        else:
            self.seq_list = None

        self.m_local_entropies = {}
        self.n_gram_counts = {}

    def __getitem__(self, idx):
        
        if self.fp is None:
            return self.seqs()[idx]
        else:
            raise NotImplementedError('Cannot index when using a filegen.')
    
    def __len__(self):
        return self.num_seqs
    
    def __iter__(self):
        return iter(self.seqs())
    
    def seqs(self):
        if self.seq_list is not None:
            return self.seq_list
        else:
            with open(self.fp, 'r', encoding='utf-8') as f:
                for line in f:
                    yield Sequence(line.strip())

    def basic_stats(self) -> dict:

        lens = sorted([len(t) for t in self.seqs()])

        tokens = sum(lens)

        if len(lens) % 2 == 0:
            median = (
                lens[len(lens) // 2] + lens[len(lens) // 2 + 1]
            ) // 2
        else:
            median = lens[len(lens) // 2 + 1]

        unique_lens = dict.fromkeys(set(lens), 0)
        for l in lens:
            unique_lens[l] += 1
        mode = -1
        max_count = -1
        for unique_len in unique_lens:
            if unique_lens[unique_len] > max_count:
                max_count = unique_lens[unique_len]
                mode = unique_len

        return {
            'token_count': tokens,
            'mean_length': tokens / len(self),
            'median_length': median,
            'mode_length': mode,
            'num_seqs': len(self)
        }

    def m_local_entropy(
        self,
        order: int = 3,
        use_gt: bool = False,
        use_cae: bool = False
    ) -> float:

        self.n_gram_counts[order] = {}
        d = self.n_gram_counts[order]

        # count all m-grams of order "order"
        for seq in self.seqs:

            s = self.example_to_str(seq)

            if len(s) < order:
                continue
            
            for i in range(order - 1, len(s)):
                context = s[i-order+1:i]
                if context in d:
                    if s[i] in d[context]:
                        d[context][s[i]] += 1
                    else:
                        d[context][s[i]] = 1
                else:
                    d[context] = {}
                    d[context][s[i]] = 1

        H = self._m_local_entropy_helper(order, use_gt, use_cae)

        self.m_local_entropies[order] = H

        return H
    
    def _m_local_entropy_helper(self, order: int, use_gt: bool, use_cae: bool) -> float:
        
        d = self.n_gram_counts[order]

        # find hapax legomena, count of m-grams first
        total_n = 0
        total_f_1 = 0
        for context in d:
            for symbol in d[context]:
                if d[context][symbol] == 1:
                    total_f_1 += 1
                total_n += d[context][symbol]
        
        ns_by_context = []
        entropies_by_context = []

        for context in d:

            # raw counts of completions for this context
            context_counts = torch.tensor(
                [
                    d[context][symbol] for symbol in d[context]
                ],
                dtype=torch.float32
            )
            n_this_context = context_counts.sum().item()

            dist = context_counts / n_this_context # normalize this context's dist to get ML estimate p^

            if use_gt: # TODO: finding that this gives a lower value, but should be higher?
                dist *= (1 - (total_f_1 / total_n)) # simplified GT smoothing to get p~
            
            if use_gt and use_cae: # must use GT to use CAE
                H_this_context = -(
                    (
                        dist * dist.log()
                    ) / (
                        1 - ((1 - dist) ** total_n)
                    )
                ).sum().item()
            else:
                H_this_context = -(
                    dist * dist.log()
                ).sum().item()
            
            ns_by_context.append(n_this_context)
            entropies_by_context.append(H_this_context)

        ns_by_context = torch.tensor(ns_by_context, dtype=torch.float32)

        dist = ns_by_context / total_n # ML estimate p^
        
        entropies_by_context = torch.tensor(entropies_by_context)

        return (dist * entropies_by_context).sum().item() # expectation

    def example_to_str(self, seq):
        return seq
    
    def excess_entropy(self):

        # Futrell and Hahn 2024. Requires entropy rate to be calculated/estimated.
        # Entropy rate for n-grams and PFSAs should be trivial.
        # Entropy and entropy rate might be trivially related.
        # Can put a unigram distribution into this and hope for 0 b/c context is useless.

        E = 0.
        order = 1

        # TODO: theoretical MLS
        h = self.grammar.entropy().item() / self.grammar.estimated_mls()

        # TODO: this is probably not right, though it seems pretty good
        while True:
            addl_entropy = self.m_local_entropy(order=order)
            if h > addl_entropy:
                break
            E += addl_entropy - h
            order += 1

        return E

    def shuffle(self, length_or_window: str, window_size: int = None):
        assert length_or_window in ('length', 'window'), 'Must give "length" or "window".'
        if length_or_window == 'window':
            assert type(window_size) == int, 'Window size must be an integer.'
            assert window_size > 1, 'Window size must be greater than 1.'

        if length_or_window == 'length':
            for i in range(len(self.seqs)):
                tokens = self.grammar.tokenize(self.seqs[i])
                Random(len(tokens)).shuffle(tokens)
                self.seqs[i] = Sequence(self.grammar.untokenize(tokens, False))
        else:
            for i in range(len(self.seqs)):
                tokens = self.grammar.tokenize(self.seqs[i])
                for j in range(0, len(tokens), window_size):
                    window = tokens[j:j + window_size]
                    Random(window_size).shuffle(window)
                    for k in range(len(window)):
                        tokens[j + k] = window[k]



class SequenceDataLoader(torch.utils.data.DataLoader):

    def __init__(
        self, 
        ds: SequenceDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        max_length: int = 128
    ):
        super().__init__(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: ds.grammar.batch_tokenize(
                x,
                return_tensors='pt',
                truncate_length=max_length
            )
        )
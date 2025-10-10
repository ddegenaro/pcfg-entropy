from time import time
from typing import Union, Iterable
from concurrent.futures import ThreadPoolExecutor

import torch

from utils import SequenceDataset

class PFSA:

    def __init__(
        self,
        num_symbols: int = 10,
        num_states: int = 10,
        chr_ord_offset: int = 97,
        file: str = None,
        device: str = None,
        seed: int = 42
    ):
        
        """
        Define a probabilistic finite-state automaton with optimizable probabilities.

        Args:
            `num_symbols`: `int` - the number of symbols in the grammar.
            `num_states`: `int` - the number of hidden states in the grammar.
            `chr_ord_offset`: `int` - constant shift to Unicode representations of integer-valued terminal symbols.
            `transitions`: `torch.Tensor | str` - probabilities for weighting the rules. Generated if not passed.
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

        self.chr_ord_offset = chr_ord_offset

        if file is None:

            self.transitions = torch.randn(
                (num_symbols, num_states, num_states + 1),
                device=device,
                generator=self.generator
            ).softmax(2)

            self.pi = torch.randn(
                (num_states,),
                device=device,
                generator=self.generator
            ).softmax(0)

            self.num_symbols: int = num_symbols
            self.num_states: int = num_states

        else:
        
            if type(file) == str:
                data = torch.load(file, map_location=device)
                self.pi = data['pi']
                self.transitions = data['transitions']
            else:
                raise TypeError(
                    f"Expected transitions to be a string or Torch tensor but got: {type(file)}"
                )

            self.num_symbols: int = self.transitions.shape[0]
            self.num_states: int = self.transitions.shape[1]
            assert self.transitions.shape[1] == self.transitions.shape[2]
            assert self.pi.shape[0] == self.transitions.shape[1]

        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

        self.states_ordered: list[int] = [
            x for x in range(self.num_states)
        ]

        self.symbols_ordered: list[str] = [
            chr(x + chr_ord_offset) for x in range(self.num_symbols)
        ]

        self.Q = set(self.states_ordered)
        assert len(self.Q) == len(self.states_ordered) == self.num_states
        self.Sigma = set(self.symbols_ordered)
        assert len(self.Sigma) == len(self.symbols_ordered) == self.num_symbols

        self.pad_id = self.num_symbols

        if not torch.allclose(self.transitions.sum(2), torch.tensor(1., device=self.device)):
            print(f'Warning: probability distributions not summing to 1.')

    def save(self, fp: str):
        torch.save(
            {
                'pi': self.pi.cpu(),
                'transitions': self.transitions.cpu()
            },
            fp
        )
        print(f'Saved pi and transitions to {fp} successfully.')

    def tokenize(self, seq: str, return_tensors=None) -> Union[list[int], torch.Tensor]:

        ids = [
            ord(x - self.chr_ord_offset) for x in seq
        ]

        if return_tensors == 'pt':
            return torch.tensor(ids, dtype=int, device=self.device)
        else:
            return ids
        
    def batch_tokenize(
        self,
        seqs: Iterable[str],
        return_tensors=None,
        truncate_length=None
    ) -> dict[str, Union[list[int], torch.Tensor]]:
        
        input_ids = [self.tokenize(t) for t in seqs]

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

        return ''.join([chr(s + self.chr_ord_offset) for s in as_list])
    
    def p_seq(self, seq): # TODO
        return 0
    
    def _generate_one(self, max_length: int) -> str: # TODO
        pass

    def generate(
        self,
        max_length: int = 128,
        num_seqs: int = 1,
        max_threads: int = None
    ) -> list[str]:
        
        if max_length is None or max_length <= 0:
            max_length = torch.inf

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [
                executor.submit(self._generate_one, max_length)
                for _ in range(num_seqs)
            ]
            seqs = [future.result() for future in futures]

        return seqs
    
    def entropy(self) -> torch.Tensor: # TODO
        
        P = self.transitions.sum(0) # sum over symbol dimension

        stop_probs = self.transitions[:, :, -1]

    def _get_steady_state(self) -> torch.Tensor: # TODO

        pass

    def optimize(
        self,
        H_t: float,
        do_logging: bool = True,
        tol: float = 1e-6,
        lr: float = 0.001,
        log_freq: int = 1000,
        max_iter: int = 100_000,
        max_time: float = 300.0,
        K: int = 1000
    ) -> tuple[torch.Tensor, float, float, list[float]]:
        
        losses = []
        
        DH = torch.tensor(H_t, dtype=torch.float32, requires_grad=False, device=self.device)
        criterion = torch.nn.MSELoss()
        if do_logging:
            print(f'criterion: {criterion.__class__.__name__}')
            print(f'Testing {K} random initializations...')

        with torch.no_grad():
            # Compute initial loss with current rules
            normalized_current_pi = self.pi.softmax(0)
            normalized_current_transitions = self.transitions.softmax(2)

            original_pi = self.pi
            original_transitions = self.transitions
            
            self.pi = normalized_current_pi
            self.transitions = normalized_current_transitions

            best_loss = criterion(self.entropy(), DH).item()
            best_pi = original_pi.clone()
            best_transitions = original_transitions.clone()  # Store the raw rules
            
            # Try K random initializations
            for k in range(K):
                # Generate random tensor of same shape (raw values)
                candidate_pi = torch.randn(
                    self.pi.shape,
                    device=self.device,
                    generator=self.generator
                )
                candidate_pi_normalized = candidate_pi.softmax(0)
                candidate_transitions = torch.randn(
                    self.transitions.shape,
                    device=self.device,
                    generator=self.generator
                )
                candidate_transitions_normalized = candidate_transitions.softmax(2)
                
                # Compute loss with candidate rules
                self.pi = candidate_pi_normalized
                self.transitions = candidate_transitions_normalized
                candidate_loss = criterion(self.entropy(), DH).item()
                
                # Update best if this is better
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_pi = candidate_pi.clone()
                    best_transitions = candidate_transitions.clone()  # Store raw values
                    if do_logging:
                        print(f'New best at initialization {k}: loss = {best_loss:.6f}')
        
        # Set rules to best found initialization
        self.pi = best_pi
        self.transitions = best_transitions
        if do_logging:
            print(f'Best initialization loss: {best_loss:.6f}')
            print('Starting optimization...')
        
        self.pi = torch.nn.Parameter(self.pi)
        self.transitions = torch.nn.Parameter(self.transitions)
        best_optimization_loss = float('inf')
        best_optimization_pi = None
        best_optimization_transitions = None
        optimizer = torch.optim.AdamW([self.pi, self.transitions], lr=lr)
        
        i = 0
        if do_logging:
            print('-----------------------------------------------------')
        start = time()
        while True:
            
            optimizer.zero_grad()
            
            normalized_pi = self.pi.softmax(0)
            normalized_transitions = self.transitions.softmax(2)

            original_pi = self.pi
            original_transitions = self.transitions

            self.pi = normalized_pi
            self.transitions = normalized_transitions

            loss = criterion(self.entropy(), DH)

            self.pi = original_pi
            self.transitions = original_transitions

            loss.backward()
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
            if best_optimization_pi is not None:
                self.pi = best_optimization_pi.softmax(0).detach()
            else:
                self.pi = self.pi.softmax(0).detach()
            if best_optimization_transitions is not None:
                self.transitions = best_optimization_transitions.softmax(2).detach()
            else:
                self.transitions = self.transitions.softmax(2).detach()
        
    def to(self, device: Union[str, torch.device]):
        self.pi = self.pi.to(device)
        self.transitions = self.transitions.to(device)
        self.device = device
        if device == 'cpu' or device == torch.device('cpu'):
            self.generator = self.cpu_generator
        else:
            self.generator = self.gpu_generator
            if self.generator is None:
                print(f'No GPU generator available - does this machine have a GPU?')
        return self
    
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

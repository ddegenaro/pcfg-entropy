import torch

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        grammar,
        num_seqs: int = 100,
        max_length: int = 128
    ):
        
        super().__init__()

        if max_length is None or max_length <= 0:
            max_length = torch.inf
        self.grammar = grammar
        self.max_length = max_length
        self.examples = self.grammar.generate(
            max_length=max_length,
            num_seqs=num_seqs
        )

        self.m_local_entropies = {}
        self.n_gram_counts = {}

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)

    def basic_stats(self) -> dict:

        lens = sorted([len(t) for t in self.examples])

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
            'mode_length': mode
        }

    def m_local_entropy(self, order: int = 3, use_gt: bool = False, use_cae: bool = False) -> float:

        # if order in self.m_local_entropies:
        #     return self.m_local_entropies[order]

        self.n_gram_counts[order] = {}
        d = self.n_gram_counts[order]

        # count all m-grams of order "order"
        for seq in self.examples:

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

        return (dist * entropies_by_context).sum() # expectation

    
    def example_to_str(self, seq):
        return seq
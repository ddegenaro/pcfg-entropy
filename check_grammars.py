import os

from pcfg import PCFG

for grammar_file in sorted(os.listdir('grammar_matrices')):

    loc = os.path.join('grammar_matrices', grammar_file)

    grammar = PCFG(rules=loc)

    entropy = grammar.entropy().item()
    targeted_entropy = float(grammar_file.split('_')[8])
    error = abs(entropy-targeted_entropy)
    if error > 1.0:
        print(grammar_file, error)
        os.remove(loc)
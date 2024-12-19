import numpy.random as random
from typing import Iterable
from copy import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Grammar:
    def __init__(self, words: Iterable[str]):
        self.words = set(words)

    def __str__(self):
        return "Mots: " + ",".join(sorted(self.words))

    def __repr__(self):
        return "Mots: " + ",".join(sorted(self.words))

    def __call__(self, number):
        return set(random.choice(list(self.words), number))

    def __sub__(self, other):
        return Grammar(self.words - other.words)

    def __add__(self, other):
        return Grammar(self.words)

    def __contains__(self, item):
        return item in self.words

    def __copy__(self):
        return Grammar(self.words.copy())

    def __len__(self):
        return len(self.words)

    def __mul__(self, other):
        n1, n2 = min(random.geometric(1 / 2), len(self)), min(random.geometric(1 / 2), len(other))
        # print(n1, n2)
        other.words = other.words.union(self(n1))
        self.words = self.words.union(other(n2))
        return


def generate_grammars(n_words, n_langs):
    lesmots = [f"mot_{i}" for i in range(n_words)]
    basegrammars = list(
        Grammar(lesmots[(i - 1) * n_words // n_langs: i * n_words // n_langs]) for i in range(1, n_langs)
        )
    basegrammars.append(Grammar(lesmots[(n_langs - 1) * n_words // n_langs:]))
    return basegrammars

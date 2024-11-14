import numpy.random as random
from typing import Iterable
from copy import copy


class Grammar:
    def __init__(self, words: Iterable[str]):
        self.words = set(words)

    def __call__(self, number):
        return set(random.choice(list(self.words), number))

    def __sub__(self, other):
        return Grammar(self.words - other.words)

    def __copy__(self):
        return Grammar(self.words.copy())

    def __len__(self):
        return len(self.words)

    def __mul__(self, other):
        n1, n2 = min(random.geometric(1 / 2), len(self)), min(random.geometric(1 / 2), len(other))
        other.words.union(self(n1))
        self.words.union(other(n2))
        return




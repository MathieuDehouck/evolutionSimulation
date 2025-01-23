from typing import TypeVar, Iterable
import numpy as np
import random
from copy import copy

T = TypeVar('T')

class Language: 
    def __init__(self, grammar: T = None): 
        self.grammar = grammar
    
    def __sub__(self, other):
        return 1

    def __add__(self, other):
        return Language(grammar=self.grammar + other.grammar)

    def __mul__(self, other):
        g1, g2 = self.grammar, other.grammar
        if g1 is not None and g2 is not None:
            _ = g1 * g2
        return g1, g2

    def __repr__(self): 
        return repr(self.grammar)

    def __str__(self): 
        return str(self.grammar)

    def __abs__(self): 
        return Language(grammar=copy(self.grammar))

    def __copy__(self):
        return Language(grammar=copy(self.grammar))


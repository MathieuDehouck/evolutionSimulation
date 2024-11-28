import numpy.random as random
from typing import Iterable
from copy import copy
import numpy as np
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


def mean_observable(tree_list, baselangs):
    result = np.zeros((len(tree_list), len(baselangs)))
    for i, t in enumerate(tree_list):
        leaves = t.leaves()
        n_words = 0
        for leaf in leaves:
            lang = leaf.lang
            for w in lang.grammar.words:
                index = next(k for k, v in enumerate(baselangs) if w in v)
                result[i][index] += 1
                n_words += 1
        s = sum(result[i])
        for w in range(len(tree_list)):
            result[i][w] = round(result[i][w] / s, 3)
    return result


def histogram(tree_list, baselangs):
    values = mean_observable(tree_list, baselangs)
    langs = [f"Base = ${i}$" for i in range(len(baselangs))]
    x = np.arange(len(langs))
    width = .9/len(langs)
    mul = 0

    fig, ax = plt.subplots(layout='constrained')
    for baselang, arrivals in enumerate(values):
        offset = width*mul
        rects = ax.bar(x + offset, arrivals, width, label=baselang)
        ax.bar_label(rects, padding=3)
        mul += 1

    ax.set_ylabel("Proportion of taken words")
    ax.set_xticks(x + width, langs)
    ax.legend(loc='upper left', ncols=len(langs))
    ax.set_ylim(0, 1.1)
    ax.set_title("Per")
    plt.show()


if __name__ == '__main__':
    import src.worldGeometry.tree_gen as tg
    import src.worldGeometry.real_space as real_space
    n = 24
    lesmots = [f"mot_{i}" for i in range(n)]
    g1, g2, g3 = Grammar(lesmots[:n // 3]), Grammar(lesmots[n // 3:2 * n // 3]), Grammar(lesmots[2 * n // 3:])
    cube_langs = [real_space.Language([1, 0, 0], grammar=g1), real_space.Language([0, 1, 0], grammar=g2),
                  real_space.Language([0, 0, 1], grammar=g3)]
    tl, cl, ll = tg.naive_parallel_evolution(10, 5, cube_langs, beta=3/4)
    basegrammars = [g1, g2, g3]
    histogram(tl, basegrammars)
    # top_down_random_tree(10, 10, real_space.Language([0, 0, 0]))

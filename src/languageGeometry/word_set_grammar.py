import numpy.random as random
from typing import Iterable
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from numba.cpython.builtins import max_iterable


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


def sanitize_number(f):
    return round(f, 3)

def mean_observable(tree_list, baselangs):
    result = np.zeros((len(tree_list), len(baselangs)))
    for origin, t in enumerate(tree_list):
        leaves = t.leaves()
        n_words = 0
        for leaf in leaves:
            lang = leaf.lang
            for w in lang.grammar.words:
                arrival = next(k for k, v in enumerate(baselangs) if w in v)
                result[origin][arrival] += 1
                n_words += 1
        s = sum(result[origin])
        for w in range(len(tree_list)):
            result[origin][w] = sanitize_number(result[origin][w] / s)
    return result


def histogram(tree_list, baselangs, savepath=None, beta=0, mt=0, mw=0):
    values = mean_observable(tree_list, baselangs)
    langs = [f"Base = ${i}$" for i in range(len(baselangs))]
    x = np.arange(len(langs))
    width = .9 / len(langs)
    mul = 0

    # dist_matrix = np.zeros((len(baselangs), len(baselangs)))
    # for i in range(len(baselangs)):
    #     for j in range(len(baselangs)):
    #         dist_matrix[i, j] = baselangs[i] - baselangs[j]

    # print(dist_matrix)

    fig, ax = plt.subplots(layout='constrained')
    for baselang, arrivals in enumerate(values.T):
        offset = width * mul
        rects = ax.bar(x + offset, arrivals, width, label=baselang)
        ax.bar_label(rects, padding=3)
        mul += 1

    ax.set_ylabel("Proportion of taken words")
    ax.set_xticks(x + width, langs)
    ax.legend(loc='upper left', ncols=len(langs))
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Word switching for beta = {beta},\nn_langs = {len(langs)}, t = {mt}, w = {mw}")
    if savepath is None:
        plt.show()
    else:
        plt.savefig(f"{savepath}.pdf", format="pdf")


if __name__ == '__main__':
    from sys import path
    path.append("../..")
    path.append("../../..")
    path.append("..")
    import src.worldGeometry.tree_gen as tg
    import src.worldGeometry.real_space as real_space

    n = 24
    n_langs = 4
    max_time = 10
    max_width = 9

    cube_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    lesmots = [f"mot_{i}" for i in range(n)]
    basegrammars = list(Grammar(lesmots[(i - 1) * n // n_langs : i * n // n_langs]) for i in range(1, n_langs))
    basegrammars.append(Grammar(lesmots[(n_langs - 1) * n // n_langs:]))
    cube_langs = list(real_space.Language(coordinates=cube_vertices[i], grammar=basegrammars[i]) for i in range(n_langs))

    # b = 0.0
    # tl, cl, ll = tg.naive_parallel_evolution(max_time, max_width, cube_langs, beta=b)
    # histogram(
    #     tl, basegrammars,
    #     savepath=f"../../Figures/WordSet/HWS_b={round(b, 2)}_nl={len(basegrammars)}_t={max_time}_w={max_width}", beta=b
    #     )

    for k, b in enumerate(np.arange(start=0, stop=1, step=.1)):
        tl, cl, ll = tg.naive_parallel_evolution(max_time, max_width, cube_langs, beta=b)
        histogram(tl, basegrammars, savepath=f"../../Figures/WordSet/HWS_b={round(b, 2)}_nl={len(basegrammars)}_t={max_time}_w={max_width}", beta=b, mt=max_time, mw=max_width)


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import chain, combinations
import tqdm
import random

DATA_PATH = "../Data/xmls"


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
                try:
                    arrival = next(index for index, v in enumerate(baselangs) if w in v.grammar)
                except StopIteration as exc:
                    print(w)
                    raise StopIteration from exc
                result[origin][arrival] += 1
                n_words += 1
        s = sum(result[origin])
        for w in range(len(tree_list)):
            result[origin][w] = sanitize_number(result[origin][w] / s)
    return result


def histogram(tree_list, baselangs, savepath=None, beta=0, mt=0, mw=0, n_words=0, labels=None):
    if labels is None:
        labels = []
    n_baselangs = len(baselangs)
    values = mean_observable(tree_list, baselangs)
    langs = [f"Base = ${i}$" for i in range(n_baselangs)]
    x = np.arange(len(langs))
    width = .9 / len(langs)
    mul = 0

    # cmap = mpl.colors.Colormap("viridis")
    dist_matrix = np.zeros((n_baselangs * n_baselangs, 3))
    for i in range(n_baselangs):
        for j in range(n_baselangs):
            dist_matrix[i * n_baselangs + j] = i, j, baselangs[i] - baselangs[j]

    fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained')
    for baselang, arrivals in enumerate(values.T):
        offset = width * mul
        rects = ax1.bar(x + offset, arrivals, width, label=baselang)
        ax1.bar_label(rects, padding=3)
        mul += 1

        ax1.set_ylabel("Proportion of taken words")
        ax1.set_xticks(x + width, langs)
        ax1.legend(loc='upper left', ncols=len(langs))
        ax1.set_ylim(0, 1.1)
        ax1.set_title(f"Word switching for beta = {sanitize_number(beta)},\nn_langs = {len(langs)}, t = {mt}, w = {mw}, n_words={n_words}")

    scat = ax2.scatter(dist_matrix[:, 0], dist_matrix[:, 1], c=dist_matrix[:, 2])
    ax2.set_title("Distance between $i$ and $j$")
    ax2.set_ylabel("$j$")
    ax2.set_xlabel(", ".join(f"${i}$ - {v}" for i, v in enumerate(labels)))
    ax2.set_xticks(x)
    ax2.set_yticks(x)
    fig.colorbar(scat, ax=ax2)

    if savepath is None:
        plt.show()
    else:
        plt.savefig(f"{savepath}.pdf", format="pdf")


def enumerate_histograms():
    n = 50
    n_langs = 2
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

    sphere_vertices = [
        ([49, 3], "Paris"),
        ([52.5, 13.4], "Berlin"),
        ([51.5, -0.12], "Londres"),
        ([40.4, -3.7], "Madrid"),
        ([41.9, 12.5], "Rome"),
        ([55.7, 12.6], "Copenhague"),
        ([40.7, -73.9], "New York"),
    ]
    lesmots = [f"mot_{i}" for i in range(n)]
    basegrammars = list(ws.Grammar(lesmots[(i - 1) * n // n_langs: i * n // n_langs]) for i in range(1, n_langs))
    basegrammars.append(ws.Grammar(lesmots[(n_langs - 1) * n // n_langs:]))

    cube_langs = list(
        real_space.Language(coordinates=cube_vertices[i], grammar=basegrammars[i]) for i in range(n_langs)
    )
    europe_langs = list(
        sphere.Language(coordinates=sphere_vertices[i][0], grammar=basegrammars[i]) for i in range(n_langs)
    )

    for k, b in enumerate(np.arange(start=0, stop=1, step=.1)):
        tl, cl, ll = tg.naive_parallel_evolution(max_time, max_width, europe_langs, beta=b)
        histogram(
            tl, europe_langs,
            savepath=f"../Figures/WordSet/HWS_Sphere_b={round(b, 2)}_nw={n}_nl={len(basegrammars)}"
                     f"_t={max_time}_w={max_width}",
            beta=b, mt=max_time, mw=max_width, n_words=n,
            labels=[v[1] for i, v in enumerate(sphere_vertices) if i < n_langs]
            )


def loan_matrix(leaves):
    result_mat = np.zeros((len(leaves), len(leaves)))
    for (i, l1), (j, l2) in combinations(enumerate(leaves), 2):
        result_mat[i][j] = len(l1.words.intersection(l2)) / len(l1)
        result_mat[j][i] = len(l2.words.intersection(l1)) / len(l2)
    return result_mat


def loss(m1, m2):
    return sum((m1 - m2)**2)


def evolve(initial_conditions):
    goal, n_words, n_langs, base_vertices = initial_conditions["goal"], initial_conditions["n_words"], initial_conditions["n_langs"], initial_conditions["base_vertices"]
    with open(goal) as f:
        m1 = pd.read_csv(f)
    print(m1.to_array())
    max_width = len(m1[0][1:]) / n_langs  # n_langs should be a divisior of the number of modern languages studied.
    base_grammars = ws.generate_grammars(n_words, n_langs)
    base_langs = list(
        sphere.Language(coordinates=base_vertices[i], grammar=base_grammars[i]) for i in range(n_langs)
    )
    # Use https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    a, b, loss_value = initial_conditions["alpha_0"], initial_conditions["beta_0"], float("inf")
    iterations = 0
    while loss_value > initial_conditions["objective"]:
        _, _, ll = tg.enforced_naive_parallel_evolution(max_time=initial_conditions["epochs"],
                                                        max_width=max_width,
                                                        root_langs=base_langs,
                                                        alpha=a,
                                                        beta=b,
                                                        )
        res_mat = loan_matrix(ll)
        loss_value = loss(m1, res_mat)
        iterations += 1
        # Implement Gradient Estimate (Only beta is a parameter for now)
        a, b = random.random(), random.random()


if __name__ == '__main__':
    from sys import path
    path.append("../..")
    path.append("../../..")
    path.append("..")
    import src.languageGeometry.word_set_grammar as ws
    import src.tree_gen as tg
    from src.worldGeometry import real_space
    from src.worldGeometry import sphere

    # enumerate_histograms()

    initialiser = {
        "epochs": 1000,
        "tree_width": None,  # Determined by Goal
        "n_words": 10000,
        "n_langs": 2,  # Determined by the number of .xml used to create goal
        "alpha_0": None,  # Not used for now
        "beta_0": .5,  # Collision Factor
        "goal": f"{DATA_PATH}/../west_europe_modern_mat.csv",
        "moderns": ["French", "English", "Italian", "German", "Spanish", "Dutch", "Danish"],
        "base_vertices": [
            ([49, 3], "Paris"),
            ([52.5, 13.4], "Berlin"),
        ],
        "objective": 1,
    }

    evolve(initialiser)

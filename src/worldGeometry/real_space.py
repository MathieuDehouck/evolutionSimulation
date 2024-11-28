from typing import TypeVar, Iterable
import numpy as np
import tree_gen
import random
import matplotlib.pyplot as plt
from copy import copy

T = TypeVar('T')


class Language:
    def __init__(self,
                 coordinates: Iterable[float],
                 grammar: T = None):
        self.coordinates = np.array(coordinates)
        self.grammar = grammar

    @property
    def dim(self):
        return len(self.coordinates)

    def __sub__(self, other):
        def function(e):
            x, y = e
            return (x - y) ** 2

        return sum(map(function, zip(self.coordinates, other.coordinates)))

    def __add__(self, other):
        def function(e):
            x, y = e
            return (x + y) / 2

        return Language(coordinates=function((self.coordinates, other.coordinates)), grammar=self.grammar + other.grammar)

    def __mul__(self, other):
        g1, g2 = self.grammar, other.grammar
        if g1 is not None and g2 is not None:
            _ = g1 * g2
        return g1, g2

    def __repr__(self):
        return repr(self.coordinates) + " --- " + repr(self.grammar)

    def __str__(self):
        return str(self.coordinates) + " --- " + str(self.grammar)

    @staticmethod
    def ambient_space():
        return plt.axes(projection='3d')

    def __call__(self, vertices, edges, **kwargs):
        """
        Parameters
        ----------
        vertices: Dict with keys ids of languages with colors
        edges: List of pairs of languages with the language they derive from
        axes: Pyplot axes to plot in. If None, will default to self.ambient_space()

        Returns
        -------
        Plot of a graph
        """
        assert self.dim == 3
        if kwargs.get('axes', None) is None:
            axes = self.ambient_space()
        else:
            axes = kwargs['axes']

        vertices = dict(vertices)
        points = list(t[0] for t in vertices.values())
        coordinates = tuple(np.array(list(t[i] for t in points)) for i in range(3))
        colors = list(map(lambda t: t[1], vertices.values()))
        axes.scatter(*coordinates, c=colors)
        for (i, j) in edges:
            p1, p2 = vertices[i][0], vertices[j][0]
            edge_coordinates = tuple([p1[i], p2[i]] for i in range(3))
            edge_color = vertices[j][1]
            axes.plot(*edge_coordinates, color=edge_color)

        collision_list = kwargs['collision_list']
        if collision_list is not None:
            for c in collision_list:
                p1, p2, edge_color = c
                edge_coordinates = tuple([p1[i], p2[i]] for i in range(3))
                axes.plot(*edge_coordinates, color=edge_color, linestyle='dashed')

        return axes

    def __getitem__(self, item):
        return self.coordinates[item]

    def __abs__(self):
        i = random.randint(0, self.dim - 1)
        m = random.choice((-2, -1, 1, 2))
        coordinages = self.coordinates.copy()
        coordinages[i] += m
        grammar = copy(self.grammar)
        return Language(coordinages, grammar=grammar)

    def __copy__(self):
        return Language(coordinates=copy(self.coordinates), grammar=copy(self.grammar))


if __name__ == '__main__':
    lang0 = Language([0, 0, 0])
    # print(lang0)
    evo = tree_gen.top_down_random_tree(10, 10, root_lang=lang0)
    # print(str(evo))
    evo.plot()

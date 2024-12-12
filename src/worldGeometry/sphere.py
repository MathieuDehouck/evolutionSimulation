import random
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

class Language:
    def __init__(self, coordinates: [float, float], grammar: "T" = None):
        assert len(coordinates) == 2
        self.coordinates = coordinates  # Latitude and Longitude
        self.grammar = grammar

    def __sub__(self, other):
        lambda_a, phi_a = self[:]
        lambda_b, phi_b = other[:]
        return np.arccos(np.sin(phi_a)*np.sin(phi_b) + np.cos(phi_a)*np.cos(phi_b)*np.cos(lambda_b - lambda_a))

    def __add__(self, other):
        def arithmean(e):
            x, y = e
            return (x + y) / 2

        return Language(tuple(arithmean((np.array(self[:]), np.array(other[:])))), grammar=self.grammar + other.grammar)

    def __mul__(self, other):
        g1, g2 = self.grammar, other.grammar
        if g1 is not None and g2 is not None:
            _ = g1 * g2
        return g1, g2

    def __repr__(self):
        return repr(self.coordinates)

    def __str__(self):
        return f"Latitude: {self[0]} --- Longitude: {self[1]}"

    @staticmethod
    def ambient_space():
        r = 1
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        axes = plt.axes(projection='3d')
        axes.plot_surface(
            x, y, z, rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0
        )
        return axes

    def lat_lon_to_3D(self, e):
        lat, lon = e
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        return np.array([x, y, z])

    def __call__(self, vertices, edges, **kwargs):
        """
        Parameters
        ----------
        vertices: Dict with keys ids of languages with colors
        edges: List of pairs of languages with the language they derive from

        Returns
        -------
        Plot of a graph
        """
        axes = self.ambient_space()
        vertices = list(map(lambda t: t[1],
                            sorted(vertices, key=lambda t: t[0])))
        # List of the latitude + longitude of the points
        _points = list((t[0] for t in vertices))
        points = list(map(self.lat_lon_to_3D, _points))
        # List of all coords for each axis
        coordinates = tuple(np.array(list(t[i] for t in points))
                            for i in range(3))
        colors = list(map(lambda t: t[1], vertices))
        axes.scatter(*coordinates, c=colors)
        for (i, j) in edges:
            # edges[0] = (0, 0) since the first element in depth first order
            # is the root
            p1, p2 = points[i], points[j]
            edge_coordinates = tuple([p1[i], p2[i]] for i in range(3))
            edge_color = colors[j]
            axes.plot(*edge_coordinates, color=edge_color)
        return axes

    def __getitem__(self, item):
        return self.coordinates[item]

    def __abs__(self):
        i = random.randint(0, 1)
        m = np.random.choice((-np.pi/6, np.pi/6, np.pi/4, -np.pi/4))
        coordinages = list(self[:])
        coordinages[i] += m
        coordinages = tuple(coordinages)
        return Language(coordinages, grammar=copy(self.grammar))

    def __copy__(self):
        return Language(coordinates=copy(self.coordinates), grammar=copy(self.grammar))



if __name__ == '__main__':
    import tree_gen
    lang0 = Language((0, 0))
    # print(lang0)
    evo = tree_gen.top_down_random_tree(10, 10, root_lang=lang0)
    # print(str(evo))
    axes = evo.plot()
    plt.show()

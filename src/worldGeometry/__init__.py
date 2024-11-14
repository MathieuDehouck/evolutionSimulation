"""
Contains Classes to Build Random Trees based on Different Space Geometries/Representations of the World.
It does not yet take language proximity into account.

A World Geometry should be a class `Language` with the following methods defined:
    __add__(self, other) returns an instance of Language between the points
    __sub__(self, other) returns the distance between two instances of Language in the world
    __str__ and __repr__ for debugging
    __call__(vertices, edges, *args, *kwargs) which plots a tree based on a list of its vertices (with colours) and
        edges.
    random_modif(self) returns an instance of Language with a small modification on the position of its argument
"""
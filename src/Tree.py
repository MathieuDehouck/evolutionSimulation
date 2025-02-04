"""
..
  Tree related classes and functions
  Mathieu Dehouck
  10/2023
"""

from random import choice


class Node():
    cnt = 0

    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.index = self.cnt
        Node.cnt += 1

        if self.parent == None:
            self.depth = 0

        else:
            self.depth = self.parent.depth + 1
            self.parent.children.append(self)

        self.change_type = None
        self.change = None
        self.lang = None

    def __str__(self):
        return '(' + ','.join([str(ch) for ch in self.children]) + ')'

    def __repr__(self):
        return str(self.index)

    def is_leaf(self):
        """
        :returns: ``len(self.children) == 0``
        """
        return len(self.children) == 0

    def is_root(self):
        """
        :returns: ``self.parent == None``
        """
        return self.parent == None

    def _depth_first(self):
        sub = []
        [sub.extend(ch._depth_first()) for ch in self.children]
        return [self] + sub


class Tree():

    def __init__(self, root):
        self.root = root

    def breadth_first(self):
        """
        :returns: the list of node of this tree in breadth first order.
        """
        nodes = [self.root]
        raise NotImplementedYet

    @property
    def depth_first(self):
        """
        :returns: the list of node of this tree in depth first order.
        """
        return self.root._depth_first

    def leaves(self):
        """
        :returns: the list of leaf node of this tree.
        """

        return [n for n in self.root._depth_first if n.is_leaf()]


def random_tree(max_depth, max_width):
    """
    Generates a random empty tree structure.
    :param max_depth:
    :param max_width:
    :returns: a random tree with the desired shape parameters.
    """

    root = Node()
    tree = Tree(root)

    node = root
    for i in range(max_depth - 1):
        node = Node(parent=node)

    for w in range(max_width - 1):
        nodes = tree.depth_first
        nodes = [n for n in nodes if not (n.is_leaf() or n.is_root())]

        node = choice(nodes)
        for i in range(max_depth - node.depth - 1):
            node = Node(parent=node)

    return tree

import queue
import random
from random import choice
from typing import TypeVar, Generic, List

import numpy as np
from copy import copy
from matplotlib import pyplot as plt
import tqdm

T = TypeVar('T')

pcbar = ['violet', 'lavender', 'pink', 'magenta', 'red', 'peach', 'orange', 'gold',
         'yellow', 'chartreuse', 'green', 'turquoise', 'light blue', 'royal blue', 'dark blue', 'deep purple']
cmap = list(f"xkcd:{t}" for t in pcbar)


class NotImplementedYet(Exception):
    content = "BUG"


# In the following functions, language should contain a representation for a geometry and a grammar.
def cross_modification(language1, language2):
    language1 * language2


def distance(language1, language2):
    return language1 - language2


def random_modification(language):
    return abs(language)


def filiation(language1, language2):
    return language1 + language2


class Node(Generic[T]):
    id = 0

    def __init__(self, parent: "Node[Generic[T]]" = None, lang: T = None) -> None:
        self.parent: Node[Generic[T]] = parent
        self.children = []
        self.index = self.id
        Node.id += 1

        if self.parent is None:
            self.depth = 0

        else:
            self.depth = self.parent.depth + 1
            self.parent.children.append(self)

        self.change_type = None
        self.change = None
        self.lang: T = lang

    def __str__(self) -> str:
        return str(self.lang) + '\n(' + ','.join([str(ch) for ch in self.children]) + ')'

    def __repr__(self) -> str:
        return str(self.index) + ' ' + str(self.depth)

    @property
    def is_leaf(self) -> bool:
        """
        :returns: ``len(self.children) == 0``
        """
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """
        :returns: ``self.parent == None``
        """
        return self.parent is None

    @property
    def depth_first(self) -> "List[Node[Generic[T]]]":
        sub = []
        [sub.extend(ch.depth_first) for ch in self.children]
        return [self] + sub

    def __eq__(self, other):
        return self.index == other.index

    def __sub__(self, other):
        return self.lang - other.lang

    def __hash__(self):
        return hash(self.index)


class Tree(Generic[T]):
    def __init__(self, root):
        self.root = root

    def __repr__(self):
        return "\n".join(map(lambda t: '·' + repr(t) + ' ' + str(t.depth), self.depth_first))

    @property
    def breadth_first(self):
        """
        :returns: the list of node of this tree in breadth first order.
        """
        q = queue.Queue()
        q.put((self.root, 0), block=False)
        seen = set()
        nodes = [(self.root, 0)]
        while True:
            try:
                s, depth = q.get(block=False)
            except queue.Empty:
                break
            nodes.append((s, depth))
            for t in s.children:
                if t not in seen:
                    q.put((t, depth + 1))
                    seen.add(t)
        return nodes

    @property
    def depth_first(self) -> "List[Node[Generic[T]]]":
        """
        :returns: the list of node of this tree in depth first order.
        """
        return self.root.depth_first

    def leaves(self) -> "List[Node[Generic[T]]]":
        """
        :returns: the list of leaf node of this tree.
        """
        return [n for n in self.root.depth_first if n.is_leaf]

    def __str__(self):
        return "\n".join(map(lambda t: '->' * t.depth + str(t.lang), self.depth_first))

    def plot(self, colormap=None, axes=None, collision_list=None):
        if colormap is None:
            colormap = cmap
        depth = self.depth_first
        vertices = list(map(lambda t: (t.index, (t.lang, cmap[t.depth % len(cmap)])), depth))
        edges = [(t.index, t.parent.index) if not t.is_root else (t.index, t.index) for t in depth]
        if collision_list is not None:
            collision_list = [(c[0], c[1], cmap[c[2] % len(cmap)]) for c in collision_list]
        return self.root.lang(vertices, edges, axes=axes, collision_list=collision_list)


def top_down_random_tree(max_depth, max_width, root_lang):
    """
    Generates a random empty tree structure.
    :param max_depth:
    :param max_width:
    :param root_lang:
    :returns: a random tree with the desired shape parameters.
    """
    root = Node(lang=root_lang)
    tree = Tree(root)

    node = root
    for i in range(max_depth - 1):
        node = Node(parent=node)
        node.lang = random_modification(node.parent.lang)

    for w in range(max_width - 1):
        nodes = tree.depth_first
        nodes = list(filter(lambda t: not (t.is_leaf or t.is_root), nodes))
        node = choice(nodes)
        # print(node.lang)

        for i in range(max_depth - node.depth - 1):
            child = node.children
            if child:
                # Choose a child with probability.
                p_child = np.array(list(map(lambda n: 1 / distance(node, n), child)))
                s = sum(p_child)
                if s:
                    p_child = p_child / s
                else:
                    p_child = np.array([1 / len(p_child) for _ in p_child])
                a_lang = np.random.choice(child, p=p_child).lang
            else:
                a_lang = abs(node.lang)
            node = Node(parent=node)
            # + should take into account the distance of points.
            node.lang = random_modification(filiation(a_lang, node.parent.lang))

    return tree


def naive_parallel_evolution(max_time: int, max_width: int, root_langs: List[T],
                             alpha: float = 2 / 3, beta: float = 1 / 3,
                             colormap=None):
    if colormap is None:
        colormap = cmap
    tree_list = [Tree(Node(lang=lang)) for lang in root_langs]
    collision_list = []
    # Contains a list of the nodes might evolve
    leaf_list = [t.root for t in tree_list]

    for time in tqdm.trange(max_time, desc="Generating Tree"):
        random.shuffle(leaf_list)
        new_leaf_list = []

        leaf_index = 0
        while leaf_index < len(leaf_list):
            cur_lang = leaf_list[leaf_index]
            random_factor = np.random.random()

            # The further we are from max_width
            # the more chances we have of seeing evolutions.
            # This is a pure evolution.
            # Evolve base on distance
            if random_factor > beta:
                # Compute choice probabilities based on distance
                p_leaves = np.array(
                    list(
                        map(
                            lambda n: 1 / distance(cur_lang, n) # change for 1/len(leaf_list)
                            if distance(cur_lang, n) else 0.,
                            leaf_list
                        )
                    )
                )
                if np.nan in p_leaves:
                    print(p_leaves)
                s = sum(p_leaves)
                if s:
                    p_leaves = p_leaves / s
                else:
                    p_leaves = np.array([1 / len(p_leaves) for _ in p_leaves])
                a_lang = np.random.choice(leaf_list, p=p_leaves)

                # Create new children if they don't exist
                if cur_lang.children:
                    n1 = cur_lang.children[0]
                else:
                    n1 = Node(parent=cur_lang)
                    n1.lang = copy(cur_lang.lang)

                if a_lang.children:
                    n2 = a_lang.children[0]
                else:
                    n2 = Node(parent=a_lang)
                    n2.lang = copy(a_lang.lang)

                cross_modification(n1.lang, n2.lang)
                # Time should be seen as the depth in the tree
                collision_list.append((a_lang.lang, cur_lang.lang, time))
                new_leaf_list.append(n1)
                new_leaf_list.append(n2)

            else:
                if cur_lang.children:
                    pass
                else:
                    new_node = Node(parent=cur_lang)
                    new_node.lang = cur_lang.lang
                    new_leaf_list.append(new_node)

            leaf_index += 1

        leaf_index = 0
        while leaf_index < len(leaf_list):
            cur_lang = leaf_list[leaf_index]
            random_factor = np.random.random()
            if random_factor > alpha * len(new_leaf_list) / max_width:
                node = Node(parent=cur_lang)
                node.lang = random_modification(cur_lang.lang)
                new_leaf_list.append(node)
            leaf_index += 1

        leaf_list = new_leaf_list
    return tree_list, collision_list, leaf_list


def enforced_naive_parallel_evolution(max_time: int, max_width: int, root_langs: List[T],
                             alpha: float = 2 / 3, beta: float = 1 / 3,
                             colormap=None):
    if colormap is None:
        colormap = cmap
    tree_list = [Tree(Node(lang=lang)) for lang in root_langs]
    collision_list = []
    # Contains a list of the nodes might evolve
    leaf_list = [t.root for t in tree_list]

    for time in range(max_time):
        random.shuffle(leaf_list)
        new_leaf_list = []

        leaf_index = 0
        while leaf_index < len(leaf_list):
            cur_lang = leaf_list[leaf_index]
            random_factor = np.random.random()

            # The further we are from max_width
            # the more chances we have of seeing evolutions.
            # This is a pure evolution.
            # Evolve base on distance
            if random_factor > beta:
                # Compute choice probabilities based on distance
                p_leaves = np.array(
                    list(
                        map(
                            lambda n: 1 / distance(cur_lang, n) # change for 1/len(leaf_list)
                            if distance(cur_lang, n) else 0.,
                            leaf_list
                        )
                    )
                )
                if np.nan in p_leaves:
                    print(p_leaves)
                s = sum(p_leaves)
                if s:
                    p_leaves = p_leaves / s
                else:
                    p_leaves = np.array([1 / len(p_leaves) for _ in p_leaves])
                a_lang = np.random.choice(leaf_list, p=p_leaves)

                # Create new children if they don't exist
                if cur_lang.children:
                    n1 = cur_lang.children[0]
                else:
                    n1 = Node(parent=cur_lang)
                    n1.lang = copy(cur_lang.lang)

                if a_lang.children:
                    n2 = a_lang.children[0]
                else:
                    n2 = Node(parent=a_lang)
                    n2.lang = copy(a_lang.lang)

                cross_modification(n1.lang, n2.lang)
                # Time should be seen as the depth in the tree
                collision_list.append((a_lang.lang, cur_lang.lang, time))
                new_leaf_list.append(n1)
                new_leaf_list.append(n2)

            else:
                if cur_lang.children:
                    pass
                else:
                    new_node = Node(parent=cur_lang)
                    new_node.lang = cur_lang.lang
                    new_leaf_list.append(new_node)

            leaf_index += 1

        leaf_index = 0
        while leaf_index < len(leaf_list):
            cur_lang = leaf_list[leaf_index]
            if len(new_leaf_list) <= max_width * len(tree_list):
                node = Node(parent=cur_lang)
                node.lang = random_modification(cur_lang.lang)
                new_leaf_list.append(node)
            leaf_index += 1

        leaf_list = new_leaf_list
    return tree_list, collision_list, leaf_list


def plot_list_tree(tree_list, collision_list, colormap=None, ax=None):
    for t in tree_list[:-1]:
        ax = t.plot(axes=ax, colormap=colormap)
    ax = tree_list[-1].plot(axes=ax, colormap=colormap, collision_list=collision_list)
    return ax


def lang_list_tree(tree_list):
    for i, t in enumerate(tree_list):
        print(f"\nArbre {i}:\n-----------------------------------")
        print(str(t))

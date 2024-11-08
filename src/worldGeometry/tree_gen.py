import queue
from random import choice
from typing import TypeVar, Generic, List

import numpy as np

T = TypeVar('T')

pcbar = ['violet', 'lavender', 'pink', 'magenta', 'red', 'peach', 'orange', 'gold',
         'yellow', 'chartreuse', 'green', 'turquoise', 'light blue', 'royal blue', 'dark blue', 'deep purple']
cmap = list(f"xkcd:{t}" for t in pcbar)


class NotImplementedYet(Exception):
    content = "BUG"


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

    def plot(self, colormap=None):
        if colormap is None:
            colormap = cmap
        depth = self.depth_first
        vertices = list(map(lambda t: (t.index, (t.lang, cmap[t.depth % len(cmap)])), depth))
        edges = [(t.index, t.parent.index) if not t.is_root else (t.index, t.index) for t in depth]
        print(edges)
        self.root.lang(vertices, edges)


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
        node.lang = node.parent.lang.random_modif

    for w in range(max_width - 1):
        nodes = tree.depth_first
        nodes = list(filter(lambda t: not (t.is_leaf or t.is_root), nodes))
        node = choice(nodes)
        # print(node.lang)

        for i in range(max_depth - node.depth - 1):
            child = node.children
            if child:
                p_child = np.array(list(map(lambda n: node.lang - n.lang, child)))
                s = sum(p_child)
                if s:
                    p_child = p_child / s
                else:
                    p_child = np.array([1 / len(p_child) for _ in p_child])
                a_lang = np.random.choice(child, p=p_child).lang
            else:
                a_lang = node.lang.random_modif
            node = Node(parent=node)
            node.lang = (a_lang + node.parent.lang).random_modif

    return tree


def naive_parallel_evolution(max_time, max_width, root_langs: List[T]):
    tree_list = [Tree(Node(lang=l)) for l in root_langs]
    for i in range(max_time):
        continue
    return tree_list


if __name__ == '__main__':
    top_down_random_tree(10, 10, None)

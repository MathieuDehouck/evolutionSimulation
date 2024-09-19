"""
..
  Simulating (language) evolution with a tree model
  Mathieu Dehouck
  10/2023
"""

from random import choice

from Tree import random_tree

from Visualisation import toSVG


def type_tree(tree, change_types):
    """
    assigns a change type to each node in the tree
    :param tree: The tree that need to be typed
    :param change_type:
    """

    tree.root.change_type = None
    
    for node in tree.depth_first():
        if node.is_root():
            continue

        node.change_type = choice(change_types)



def evolve(language, change_map, depth=10, width=10, model='char', **kwargs):
    """
    :param depth: the maximum depth of any branch
    :param width: the maximum width of the tree
    """
    print()
    print(kwargs)

    # create a tree of te desired size
    tree = random_tree(depth, width)
    print(tree.root)
    print()

    # assign a change type to each node in the tree
    change_types = list(sorted(change_map.keys()))
    type_tree(tree, change_types)
    
    # now generate changes and evolve the languages at the same time
    tree.root.lang = language
    for node in tree.depth_first():
        if node.is_root():
            continue

        change = change_map[node.change_type].generate(node.parent.lang, {})
        node.change = change
        lang = change.affect(node.parent.lang)
        node.lang = lang

    toSVG(tree, 'test')

    return tree

"""
..
  Morphological processes
  Mathieu Dehouck
  10/2023
"""

from ..Abst_Morphology import Abstract_Flexion, Abstract_Derivation


class Redup(Abstract_Morphology):
    """
    """

    def __init__(self, pattern, change=None):
        """
        pattern represents the reduplication type ex: abc > aabc or abcabc or abcc...
        change is a potentiel sound change that applies to the reduplicated form
        """
        Abstract_Derivation.__init__(self)
        
        self.pattern = pattern
        self.change = change


    def affect(self, word):
        """
        a derivation applies to a word to create a new one.
        """
        ()



class Flexion(Abstract_Flexion):
    """
    """

    def __init__(self):
        Abstract_Flexion.__init__(self)


    def affect(self, word):
        """
        a derivation applies to a word to fill its form with a paradigm.
        """
        ()

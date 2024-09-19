"""
..
  Abstract Morphological processes
  Mathieu Dehouck
  10/2023
"""



class Abstract_Morphology(metaclass=ABCMeta):
    """
    an abstract class for the Morphology hierarchy.
    """

    @abstractmethod
    def __init__(self):
        ()




class Abstract_Derivation(Abstract_Morphology):
    """
    an abstract class for representing derivational processes.
    a derivation create new words with meanings.
    """

    @abstractmethod
    def __init__(self):
        Abstract_Morphology.__init__(self)


    @abstractmethod
    def affect(self, word):
        """
        a derivation applies to a word to create a new one.
        """
        ()



class Abstract_Flexion(Abstract_Morphology):
    """
    an abstract class for representing flexional processes.
    a flexion create new paradigm from a base form.
    """

    @abstractmethod
    def __init__(self):
        Abstract_Morphology.__init__(self)


    @abstractmethod
    def affect(self, word):
        """
        a derivation applies to a word to fill its form with a paradigm.
        """
        ()

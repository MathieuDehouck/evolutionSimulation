"""
..
  changes types that we can encounter
  Mathieu Dehouck
  10/2023
"""
from abc import ABCMeta, abstractmethod

change_types = ['Phonet']


class Abstract_Change(metaclass=ABCMeta):
    """
    an abstract class for the change hierarchy.
    """

    @abstractmethod
    def __init__(self):
        ()

    @abstractmethod
    def affect(self, language):
        """
        Apply this change to a language.
        """

    @staticmethod
    @abstractmethod
    def generate(language, args):
        """
        generate a random change of this type based on language's state and args.
        """

# some change models should also implement an initialisation function to initialise a language according to its expected structure

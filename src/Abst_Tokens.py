"""
Defines abstract concepts like Words, Forms and Senses that should be implemented according to different models.
..
  Mathieu Dehouck
  07/2024
"""
from abc import ABCMeta, abstractmethod

change_types = ['Phonet']


class Abstract_Form(metaclass=ABCMeta):
    """
    an abstract class for representing Forms.
    """

    @abstractmethod
    def __init__(self):
        ()

    @abstractmethod
    def has(self, x):
        """
        If x is found in self, return a list of pairs of indices (i, j) such that word[i:j] matches x.
        """

    @abstractmethod
    def __lt__(self, other):
        """
        Compare this word with anotger one.
        """

    @abstractmethod
    def __len__(self):
        """
        Return the length of this word.
        """

    @abstractmethod
    def __repr__(self):
        """
        Return the r of this word.
        """


class Abstract_Paradigm(metaclass=ABCMeta):
    """
    an abstract class for representing form paradigms (declension, conjugation...).
    """

    @abstractmethod
    def __init__(self, forms):
        ()



class Abstract_Meaning(metaclass=ABCMeta):
    """
    an abstract class for representing Meanings.
    """

    @abstractmethod
    def __init__(self):
        ()



class Abstract_Word(metaclass=ABCMeta):
    """
    an abstract class for representing Words.
    """

    @abstractmethod
    def __init__(self, form, meaning):
        self.form = form # can also be a paradigm
        self.meaning = meaning

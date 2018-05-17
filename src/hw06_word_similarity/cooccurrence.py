from collections import defaultdict, Counter
from scipy.sparse import lil_matrix
from functools import lru_cache
import math



def vocabulary_from_wordlist(word_list, vocab_size):
    """ Returns set of vocab_size most frequent words from a given list of words.

    >>> v = vocabulary_from_wordlist(['a','rose', 'is', 'a', 'rose', 'colour', 'rose'],2)
    >>> v == {'a', 'rose'}
    True
    >>> v = vocabulary_from_wordlist(['a','rose', 'is', 'a', 'rose'],4)
    >>> v == {'a', 'rose', 'is'}
    True
    """
    # TODO: Exercise 1
    pass


def cooccurrences(tokens, n, vocab):
    """
    This takes a list of tokens (representing a text) and returns a dictionary mapping tuples of words
    to their co-occurrence count in windows of n tokens (i.e. the maximum considered distance is n).
    In other words, for each position in the corpus, co-occurrences with n tokens to the left and to the right are
    counted. Only words in a given set of words (the vocabulary) are considered.
    (Note: co-occurrence only holds between words in different positions, not for a position with itself.)

    >>> cooccurrences(["a","rose","is","a","rose"], 2, {"rose", "a"}) == {('rose', 'a'): 3, ('a', 'rose'): 3}
    True
    >>> cooccurrences(["a","rose","is","a","rose"], 1, {"rose", "is"}) == {('rose', 'is'): 1, ('is', 'rose'): 1}
    True
    """

    # TODO insert code here
    # either from hw05_cooccurrence/cooccurrence.py or
    # from https://cla2018.github.io/cooc_func.nopy (after the deadline; May 18, 16:00)
    pass


def cooc_dict_to_matrix(cooc_dict, vocab):
    """
    This takes a dictionary (word tuples/co-occurrences -> counts) and a vocabulary;
    returns a dictionary mapping each word to an index, as well as
    a Scipy Sparse matrix containing the counts at the index positions.
    >>> d = {('rose', 'is'): 2, ('rose', 'a'): 3, ('a', 'rose'): 3, ('a', 'is'): 4, ('is', 'rose'): 5, ('is', 'a'): 6}
    >>> m, w2id = cooc_dict_to_matrix(d, {'a', 'rose', 'is'})
    >>> w2id == {'is': 1, 'a': 0, 'rose': 2}
    True
    >>> m.toarray()
    array([[ 0.,  4.,  3.],
           [ 6.,  0.,  5.],
           [ 3.,  2.,  0.]])
    >>> m.nnz
    6
    """
    word_to_id = {w: i for i, w in enumerate(sorted(vocab))}
    m = lil_matrix((len(vocab), len(vocab)))
    # TODO insert code here
    # either from hw05_cooccurrence/cooccurrence.py or
    # from https://cla2018.github.io/cooc_func.nopy (after the deadline; May 18, 16:00)
    return m, word_to_id


def ppmi_weight(cooc_matrix):
    """
    This computes a PPMI weighted version of a square matrix with non-negative elements, i.e. a new matrix is returned
    that contains for each cell of the original matrix its PPMI.

    The pointwise information is defined as:
    PMI = log( P(r,c) / (P(r)*P(c)) )
    Where r,c stand for rows and columns of the matrix and:
    P(r,c) = value_of_cell_r_c / sum_of_all_cells
    P(r) = value_of_cells_in_row_r / sum_of_all_cells
    P(c) = value_of_cells_in_column_c / sum_of_all_cells

    The PPMI keeps the positive PMI values, and replaces all negative (or undefined) values with 0.

    >>> m = lil_matrix([[1,2],[3,4]])
    >>> ppmi_weight(m).toarray()
    array([[ 0.        ,  0.10536052],
           [ 0.06899287,  0.        ]])
    """
    sum_total = cooc_matrix.sum()
    sum_in_col = cooc_matrix.sum(0).A1
    sum_in_row = cooc_matrix.sum(1).A1
    ppmi_matrix = lil_matrix(cooc_matrix.shape)
    rows, cols = cooc_matrix.nonzero()
    for row, col in zip(rows, cols):
        # TODO insert code here
        # either from hw05_cooccurrence/cooccurrence.py or
        # from https://cla2018.github.io/cooc_func.nopy (after the deadline; May 18, 16:00)
        pass
    return ppmi_matrix

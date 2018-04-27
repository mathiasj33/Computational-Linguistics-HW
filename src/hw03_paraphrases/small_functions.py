import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def square_roots(start,end,length):
    """
    Returns a 1d numpy array of the specified length, containing the square roots of equi-distant input values
    between start and end (both included).

    >>> square_roots(4,9,3)
    array([ 2.        ,  2.54950976,  3.        ])
    """
    myarray= np.linspace(start, end, num=length, endpoint=True, retstep=False)
    return np.sqrt(myarray)

    pass  # TODO: Exercise 2.1


def odd_ones_squared(rows, cols):
    """
    Returns a 2d numpy array with shape (rows, cols). The matrix cells contain increasing numbers,
    where all odd numbers are squared.

    >>> odd_ones_squared(3,5)
    array([[  0,   1,   2,   9,   4],
           [ 25,   6,  49,   8,  81],
           [ 10, 121,  12, 169,  14]])
    """
    list = [n if n % 2 == 0 else n*n for n in range(rows*cols)]
    return np.array(list).reshape(3,5)
    pass  # TODO: Exercise 2.2

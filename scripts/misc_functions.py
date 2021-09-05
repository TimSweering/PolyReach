from math import comb
from scipy.sparse import csr_matrix
import numpy as np
import time


def timeit(method):
    """

    Parameters
    ----------
    method
        Method which is timed

    Returns
    -------

    """

    # Decorator function to time other functions
    def timed(*args, **kw):
        """

        :param args:
        :param kw:
        :return:
        """
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(method.__name__ + ": " + str(te - ts))
        return result

    return timed

def timeit_measure(method):
    """

    Parameters
    ----------
    method
        Method which is timed

    Returns
    -------

    """

    # Decorator function to time other functions
    def timed(*args, **kw):
        """

        :param args:
        :param kw:
        :return:
        """
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(method.__name__ + ": start_time %f, time elapsed: %f" %(ts,(te - ts)))
        return result

    return timed


def get_carleman_to_poly(lie_list_in, sym_list):
    """
    Determines the
    Parameters
    ----------
    lie_list_in
    sym_list

    Returns
    -------

    """
    # lie_list is a list of PolyMatrix containing Lie derivatives that are used as measurement functions.
    # N amount of Lie derivatives

    # Get dimension of polynomial differential equation
    n = len(sym_list)
    max_derivative = len(lie_list_in)

    # Get maximum order of the differential equation
    max_order_observer = max([max(map(sum, lie_list_in[-1][x].monoms())) for x in range(0, n)])

    # Get order of unique monomials
    order_list = get_order_list(n, max_order_observer)

    # Length of Polyflow observer
    n_rows = n * max_derivative

    # Length of monomial list
    n_columns = comb(max_order_observer + n, max_order_observer) - 1

    # Define sparse matrix for monomial to polynomial transform matrix
    l_transform = csr_matrix((n_rows, n_columns))

    # i-th Lie derivative

    start_index = 0
    for i in range(0, max_derivative):
        # Add coefficients to transform matrix
        l_transform = find_and_assign(lie_list_in[i], order_list, l_transform, start_index)
        start_index += n

    return l_transform, order_list


def get_order_list(n, m):
    """
    Generate the order of all monomials from order 1 to order m

    Parameters
    ----------
    n : int
        The dimension of the differential equation
    m : int
        Maximum order of the monomials
    Returns
    -------
    List of
    """

    # This function gets the order of the monomials is
    base_tuple = tuple(range(0, n))
    current_tuple_list = base_tuple
    order_list = []

    # Initialize tuples for first order monomials
    for i in range(0, n):
        order_el = list((0,) * n)
        order_el[i] = 1
        order_list.append(tuple(order_el))

    # Create frequency tuples of monomials of order 2 and higher
    for _ in range(1, m):
        # initialize tuple list which describes the monomials in a polynomial
        new_tuple_list = list()

        # Get monomial which order is about to increase
        n_rows = len(current_tuple_list)
        n_col = len(base_tuple)
        print_array = np.zeros((n_rows, n_col))

        col_index = 0
        for t2 in base_tuple:
            row_index = 0

            # Get first order monomial to compare
            for t_1 in current_tuple_list:

                # The case when t_1 is only one element convert it to a tuple to be consistent
                if isinstance(t_1, int):
                    t_1 = (t_1,)
                else:
                    pass

                if t2 <= t_1[-1]:
                    new_tuple_list.append(t_1 + (t2,))
                    order_el = list((0,) * n)
                    for j in range(0, n):
                        order_el[j] = new_tuple_list[-1].count(j)
                    order_list.append(tuple(order_el))
                    print_array[row_index, col_index] = 1
                else:
                    pass
                row_index += 1
            col_index += 1

        current_tuple_list = new_tuple_list

    return order_list


def find_and_assign(current_function, order_list, l_in, start_index):
    """
    Assigns the values in front of the monomial to the appropriate element of transform matrix L

    Parameters
    ----------
    current_function : Poly
        i-th Lie derivative
    order_list : List[tuple]
        Order of monomials
    l_in : csr_matrix
        Coefficient matrix which maps monomials to polynomials
    start_index : int
        start point of Lie derivative in the linear matrix L
    Returns
    -------
    linear matrix L
    """

    # Get dimension of current Lie derivative
    n = len(current_function)

    # iterate over each element of Lie derivative
    for i in range(0, n):
        # Get monomials of Lie derivative
        monomials_i = current_function[i].monoms()

        # Get coefficients of monomials
        coefficients_i = current_function[i].coeffs()

        # Iterate over each monomial
        for j in range(0, len(monomials_i)):
            # Find the corresponding column number for the monomial
            col_number = order_list.index(monomials_i[j])

            # Update element of linear matrix L
            l_in[start_index + i, col_number] += coefficients_i[j]

    return l_in


def get_smt_equation(smt_var_list, matrix_a):
    """
    Creates equation with SMT variables
    Parameters
    ----------
    smt_var_list : SMT variables
        List of SMT variables
    matrix_a : csr_matrix
        Coefficient matrix to construct the equation


    Returns
    -------

    """
    eq_list = [0] * matrix_a.shape[0]

    def calc_smt_mat_line(k):

        for i in range(len(smt_var_list)):
            if matrix_a[k, i] == 0:
                continue
            eq_list[k] += matrix_a[k, i] * smt_var_list[i]

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.map(calc_smt_mat_line, range(matrix_a.shape[0]))
    for j in range(matrix_a.shape[0]):
        calc_smt_mat_line(j)

    return eq_list

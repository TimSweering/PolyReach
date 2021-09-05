"""
    In this file the Bernstein object is defined which bounds Polynomial functions over hyper rectangles
"""
from math import comb, factorial
import sympy as sp
import numpy as np
from sympy.printing.aesaracode import aesara_function
from sympy import Poly, Matrix


class BernsteinBound:
    """
    Bernstein bound
    """

    def __init__(self, lie_fun, lie_sym, order, time_step):
        """

        Parameters
        ----------
        lie_fun
        lie_sym
        order
        time_step
        """
        self.lie_fun = lie_fun
        self.lie_sym = lie_sym
        self.order = order
        self.time_step = time_step
        self.taylor_coefficient = self.calculate_taylor_coefficient(time_step, order)
        self.lie_fun *= self.taylor_coefficient

        self.bernstein_functions = get_bernstein_bound_function(self.lie_fun, lie_sym)

    def calculate_remainder(self, interval_list, init_guess):
        """
        Calculates the remainder around the set

        Parameters
        ----------
        interval_list
        init_guess

        Returns
        -------

        """
        n = len(interval_list)
        interval_center = np.mean(interval_list, axis=1)
        base_offset = abs(interval_list[:, 0] - interval_center)

        remainder_candidate = init_guess
        lower_bounds = [[0]] * n
        upper_bounds = [[0]] * n
        while True:
            # Update interval
            for i in range(n):
                half_width = base_offset[i].__float__() + remainder_candidate
                lower_bounds[i] = [interval_center[i] - half_width]
                upper_bounds[i] = [interval_center[i] + half_width]
            interval_bounds = lower_bounds + upper_bounds

            # get bounds
            is_valid = True
            for i in range(n):
                bernstein_bound = self.get_bound(self.bernstein_functions[i], interval_bounds)
                if self.order * bernstein_bound > remainder_candidate:
                    is_valid = False
                    break

            if is_valid:
                return remainder_candidate
            else:
                remainder_candidate *= 2

    @staticmethod
    def get_bound(bound_function, interval, buffer=None, output='abs'):
        """
        Get the bound of the polynomial based on bernstein's enclosure property

        Parameters
        ----------
        bound_function
        interval
        buffer
        output

        Returns
        -------

        """
        bernstein_coefficients = bound_function(*interval)

        if buffer is None:
            buffer = np.empty((len(bernstein_coefficients), len(bernstein_coefficients[0])), dtype=np.float64)

        np.stack(bernstein_coefficients, axis=0, out=buffer)

        if output == 'abs':
            return np.amax(np.abs(buffer), axis=0)
        else:
            return np.amin(buffer, axis=0), np.amax(buffer, axis=0)

    @staticmethod
    def calculate_taylor_coefficient(time_step, order):
        """
        Calculate Taylor's coefficient h**i /i!
        Parameters
        ----------
        time_step
        order

        Returns
        -------

        """
        return time_step ** order / factorial(order)


def tuple_from_string(string_input):
    """

    Parameters
    ----------
    string_input

    Returns
    -------

    """
    return tuple(map(int, string_input.replace('(', '').replace(')', '').split(',')))


def get_bernstein_bound_function(input_function, symbolic_array):
    """

    Parameters
    ----------
    input_function
    symbolic_array

    Returns
    -------

    """
    n = len(symbolic_array)
    coefficient_function_list = [None] * n
    max_order_list = get_max_orders(input_function)
    for i in range(n):
        monom_map = make_monom_to_int(max_order_list[i])
        mapped_symbolic_function, all_symbolics, boundary_symbols = map_unit_box_to_x([input_function[i], ],
                                                                                      symbolic_array)
        monomial_coefficients = group_monomials(mapped_symbolic_function, all_symbolics, monom_map)
        bernstein_coefficients = formulate_bernstein_coefficients(monomial_coefficients, max_order_list[i],
                                                                  boundary_symbols)

        dims_arg = dict((key, 1) for key in boundary_symbols)
        dtype_arg = dict((key, 'float64') for key in boundary_symbols)
        bernstein_mat = Matrix([bernstein_coefficient for bernstein_coefficient in bernstein_coefficients[0]])
        a2 = aesara_function(boundary_symbols, bernstein_mat, dims=dims_arg, dtypes=dtype_arg, on_unused_input='ignore')
        coefficient_function_list[i] = a2
    return coefficient_function_list


def map_unit_box_to_x(input_function_list, symbolic_array):
    """

    Parameters
    ----------
    input_function_list
    symbolic_array

    Returns
    -------

    """
    n = len(symbolic_array)
    m = len(input_function_list)
    lower_bound_list = tuple(sp.sympify([str(symbolic_array[i])[:-4] + 'l_%02d' % i for i in range(n)]))
    upper_bound_list = tuple(sp.sympify([str(symbolic_array[i])[:-4] + 'u_%02d' % i for i in range(n)]))
    boundary_symbols = lower_bound_list + upper_bound_list
    all_symbols = symbolic_array + lower_bound_list + upper_bound_list
    output_function_list = [input_function_list[i].as_expr() for i in range(m)]
    output_poly = [None] * m
    for i in range(m):
        for j in range(n):
            output_function_list[i] = output_function_list[i].subs(symbolic_array[j],
                                                                   (upper_bound_list[j] - lower_bound_list[j]) *
                                                                   symbolic_array[j] +
                                                                   lower_bound_list[j])
        output_poly[i] = Poly(output_function_list[i], all_symbols)
    return output_poly, all_symbols, boundary_symbols


def group_monomials(input_function, symbolic_array, map_dict):
    """

    Parameters
    ----------
    input_function
    symbolic_array
    map_dict
    Returns
    -------

    """

    m = len(input_function)
    output_list = [None] * m
    n = int(len(symbolic_array) / 3)
    bound_symbolics = symbolic_array[n:]

    for i in range(m):
        output_dict = {}
        monomials = input_function[i].monoms()
        coefficients = input_function[i].coeffs()
        for monom_i in monomials:
            output_dict.update({map_dict[str(monom_i[:n])]['rank']: {'tuple': np.array(monom_i[:n]), 'symbolic': 0}})
        n_monomials = len(monomials)

        total_sum = 0
        for j in range(n_monomials):
            prod_list = [bound_symbolics[k] ** monomials[j][n + k] for k in range(2 * n) if monomials[j][n + k] > 0]

            key_number = map_dict[str(monomials[j][:n])]['rank']
            if prod_list is not []:
                output_dict[key_number]['symbolic'] += coefficients[j] * np.prod(prod_list)
                total_sum = total_sum + coefficients[j] * np.prod(prod_list)
            else:
                output_dict[key_number]['symbolic'] += coefficients[j]
                total_sum = total_sum + coefficients[j]
        output_list[i] = output_dict
    return output_list


def multi_binomial(list_denominator, list_nominator):
    """

    Parameters
    ----------
    list_denominator
    list_nominator

    Returns
    -------

    """
    output = 1
    for i in range(len(list_nominator)):
        output *= comb(list_denominator[i], list_nominator[i])

    return output


def get_b_coefficient(input_dict, max_order, key_list, key):
    """

    Parameters
    ----------
    key
    input_dict
    max_order
    key_list

    Returns
    -------

    """
    candidate_keys = key_list[key_list <= key]

    total_sum = 0
    for candidate_i in candidate_keys:
        if np.any(input_dict[candidate_i]['tuple'] > input_dict[key]['tuple']):
            continue
        add1 = input_dict[candidate_i]['symbolic'] * multi_binomial(input_dict[key]['tuple'],
                                                                    input_dict[candidate_i][
                                                                        'tuple']) / multi_binomial(max_order,
                                                                                                   input_dict[
                                                                                                       candidate_i][
                                                                                                       'tuple'])
        total_sum += add1

    return total_sum


def formulate_bernstein_coefficients(input_function_matrix, max_order, _):
    """

    Parameters
    ----------
    max_order
    input_function_matrix

    Returns
    -------

    """
    m = len(input_function_matrix)
    output_list_i = []
    for i in range(m):
        current_function = input_function_matrix[i]
        keys = np.array(list(current_function.keys()))
        b_list = []
        for k in keys:
            res = get_b_coefficient(input_function_matrix[i], max_order, keys, k)
            b_list.append(res)
        output_list_i.append(b_list)

    return output_list_i


def formulate_min_max_function(input_function):
    """

    Parameters
    ----------
    input_function

    Returns
    -------

    """
    min_func_list = []
    max_func_list = []
    for coefficient_functions in input_function:
        min_func_list.append(sp.Min(*coefficient_functions))
        max_func_list.append(sp.Max(*coefficient_functions))

    return min_func_list, max_func_list


def create_aesara_function(input_function, symbolic_array):
    """

    Parameters
    ----------
    input_function
    symbolic_array

    Returns
    -------

    """
    dims_arg = dict((key, 1) for key in symbolic_array)
    dtype_arg = dict((key, 'float64') for key in symbolic_array)
    aesara_function(symbolic_array, [input_function[0]], dims=dims_arg, dtypes=dtype_arg)
    pass


def make_monom_to_int(max_tuple):
    """

    Parameters
    ----------
    max_tuple

    Returns
    -------

    """
    n = len(max_tuple)

    max_list = list(max_tuple)

    sub_lists = [list(range(max_list[i] + 1)) for i in range(n)]
    len_list = [len(sub_i) for sub_i in sub_lists]

    cum_len_list = [0] * n
    total_elements = 1
    for i in range(n):
        total_elements *= len_list[-1 - i]
        cum_len_list[-1 - i] = total_elements

    output_matrix = np.zeros((total_elements, n), dtype=np.int)
    output_matrix[:len_list[-1], -1] = sub_lists[-1]
    for i in range(1, n):
        output_matrix[:cum_len_list[-1 - i], (-1 - i):] = np.concatenate(
            (np.repeat(sub_lists[-1 - i], cum_len_list[-i]).reshape((-1, 1)),
             np.tile(output_matrix[:cum_len_list[-i], -i:], (len_list[-i - 1], 1))), axis=1)

    output_dict = {}
    counter = 0
    for row_i in range(output_matrix.shape[0]):
        output_dict.update({str(output_matrix[row_i, :]).replace('[', '(').replace(']', ')').replace(' ', ', '): {
            'rank': counter, 'tuple': output_matrix[row_i, :]}})
        counter += 1

    return output_dict


def get_max_orders(input_function):
    """

    Parameters
    ----------
    input_function

    Returns
    -------

    """
    n = len(input_function.free_symbols)
    m = len(input_function)
    monomial_lists = [input_function_i.monoms() for input_function_i in input_function]
    max_order_list = []
    for i in range(m):
        max_order_list_i = [0] * n
        for j in range(n):
            for monom_t in monomial_lists[i]:
                if monom_t[j] > max_order_list_i[j]:
                    max_order_list_i[j] = monom_t[j]
        max_order_list.append(max_order_list_i)
    return max_order_list
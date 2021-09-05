""""
docstring
"""
from typing import Tuple
from typing import List, Union
from abc import ABC
import copy
from math import comb

from typing import Type
import json
from matplotlib import pyplot as plt
import numpy as np
import numba as nb
import pypolycontain as pp
from scipy.linalg import block_diag
from scipy.optimize import linprog

from scripts.misc_functions import get_order_list
from scripts.polyflow import Domain


class BaseOperations(ABC):
    """
    BaseOperations
    Basic operation which apply to Zonotope, PolynomialZonotope and Polynomial ZonotopeList
    Such as minkowski sum and other operations which apply to all sets.
    """

    def __iadd__(self, zonotope_in):
        """
        Minkowski sum which stores the output in the self variable

        Parameters
        ----------
        zonotope_in : Zonotope
            Set which is added to the current set

        Returns
        -------
        Resulting set of the minkowski sum
        """

        self.minkowski_zonotope_addition(zonotope_in)
        return self

    exponent = None
    dimension_low = None
    is_interval = False
    is_empty = False

    def __init__(self, c, gi, **kwargs):
        """
        Constructor of the Base operations

        Parameters
        ----------
        c : ndarray
            Center of set
        gi : ndarray
            Independent generators
        kwargs
        """

        self.GI = gi
        self.c = c

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.order = self.calculate_order()

    def calculate_order(self) -> float:
        """
        Calculates the order of the set.
        order = Amount of generators / dimension of set

        Returns
        -------
        order of set
        """
        if self.GI is not None:
            return self.GI.shape[1] / self.c.shape[0]
        else:
            print('There are no generators')
            raise Exception

    def __matmul__(self, transform_matrix: np.ndarray):
        """
        Applies linear transformation to set (center + generators).
        The output is stored in the self variable
        Parameters
        ----------
        transform_matrix : np.ndarray

        Returns
        -------
        Set after transformation
        """

        self.c = transform_matrix.dot(self.c)
        if self.GI is not None:
            self.GI = transform_matrix.dot(self.GI)

        return self

    def get_center(self) -> np.ndarray:
        """
        Get center of set
        Returns
        -------

        """

        return self.c

    def get_dim_low(self) -> int:
        """
        Returns the dimension of the lower dimension R^n

        Returns
        -------
        Dimension of the lower dimension
        """

        return self.dimension_low

    def get_set_dimension(self):
        """
        Get dimension of set
        Returns
        -------

        """

        return len(self.c)

    def get_exponent(self):
        """
        Get order of monomial, which the set represents
        Returns
        -------

        """

        return self.exponent

    def get_gi(self):
        """
        Get independent generators of set

        Returns
        -------

        """

        return self.GI

    def get_order(self):
        """
        Get order of the geometry (amount of generators / dimension)
        Returns
        -------

        """

        return self.order

    def minkowski_zonotope_addition(self, zonotope_in):
        """
        Adds a Zonotope to the geometry. This addition is done by concatenating GI

        Parameters
        ----------
        zonotope_in

        Returns
        -------

        """

        self.c += zonotope_in.get_center()
        if zonotope_in.get_gi() is not None:
            if self.get_gi() is not None:
                self.GI = np.concatenate((self.get_gi(), zonotope_in.get_gi()), axis=1)
            else:
                self.GI = zonotope_in.get_gi()

    def set_dim_low(self, n_low):
        """
        Set
        Parameters
        ----------
        n_low

        Returns
        -------

        """
        self.dimension_low = n_low

    def set_exponent(self, exponent):
        """

        Parameters
        ----------
        exponent

        Returns
        -------

        """
        # Set the order, which the geometry represents
        self.exponent = exponent

    def to_info_string(self):
        """
        Creates string for the __str__ function of the objects
        An integer is converted to a string
        A numpy array is converted to a tuple, which describes the shape
        If there is no information "None" then it is shown as 'not defined'

        Returns
        -------

        """

        base_attr_list = ['c', 'GI', 'exponent', 'n_low']
        output_dict = {}
        for attr in base_attr_list:
            value = getattr(self, attr)
            if type(value) == np.ndarray:
                value = value.shape
            output_dict[attr] = '%s: %s' % (attr.ljust(8), str(value) if value is not None else 'not defined')
        return output_dict

    def get_is_empty(self) -> bool:
        """
        Checks whether the generator information is important or not.
        If the generator is important then it is True
        If only the size of the generator matrix is important then it is False.

        Returns
        -------

        """
        return self.is_empty

    def get_json_dict(self, attr_oi) -> dict:
        """
        Get dictionary variable with the "item" in json-friendly format

        Parameters
        ----------
        attr_oi : str
            Name of attribute
        Returns
        -------

        """
        try:
            variable_of_interest = getattr(self, attr_oi)
        except Exception as e:
            print(e)
            print('retrieving attribute has failed')
            raise Exception

        if attr_oi in ['c', 'GI']:
            return matrix_handler(attr_oi, variable_of_interest, self.is_empty)
        else:
            return {attr_oi: variable_of_interest}

    def combine_dict(self, attr_list: List[str]) -> dict:
        """
        Combines all dictionaries of the attributes of interest.

        Parameters
        ----------
        attr_list : List[str]
            List of attributes of interest

        Returns
        -------
        Dictionary with all attribute information 
        """
        output_dict = {}
        list_dicts = [self.get_json_dict(attr_oi) for attr_oi in attr_list]
        for dict_i in list_dicts:
            output_dict.update(dict_i)

        return output_dict


class Zonotope(BaseOperations):
    """
    Zonotope
    Class representing the point symmetric polytope zonotope.
    This class includes operations of the zonotope
    """

    def to_interval_hull(self):
        """
        Overapproximates the zonotope with an interval hull and
        returns an interval hull object

        Returns
        -------

        """
        new_gi = np.diag(np.sum(np.abs(self.GI), axis=1))
        new_center = copy.deepcopy(self.c)
        return IntervalHull(new_center, new_gi)

    def __add__(self, input_zonotope):
        """
        Addition of two zonotopes with different factors before the generators

        Parameters
        ----------
        input_zonotope : Zonotope

        Returns
        -------

        """
        gi_new = np.concatenate((self.GI, input_zonotope.GI), axis=1)
        center_new = self.c + input_zonotope.c

        return Zonotope(center_new, gi_new)

    @classmethod
    def from_interval_list(cls, interval_list: np.ndarray):
        """

        Parameters
        ----------
        interval_list : ndarray
            2D array describing the interval 

        Returns
        -------

        """

        center = np.sum(interval_list, axis=1, keepdims=True) / 2
        generators = np.diag(interval_list[:, 0] - center.flatten())
        return cls(center, generators, is_interval=True)

    def interval_coordinate(self, doi=None):
        """
        Returns the coordinates of the interval hull in the specified plane.

        Parameters
        ----------
        doi : list
            indices of the plane of interest

        Returns
        -------

        """
        if not self.is_interval:
            print('is not an interval')
            return

        if doi is None:
            doi = [0, 1]
        index_x = doi[0]
        index_y = doi[1]
        x_coordinate = np.array([self.c[index_x] + self.GI[index_x, index_x],
                                 self.c[index_x] - self.GI[index_x, index_x],
                                 self.c[index_x] - self.GI[index_x, index_x],
                                 self.c[index_x] + self.GI[index_x, index_x],
                                 self.c[index_x] + self.GI[index_x, index_x]])
        y_coordinate = np.array([self.c[index_y] + self.GI[index_y, index_y],
                                 self.c[index_y] + self.GI[index_y, index_y],
                                 self.c[index_y] - self.GI[index_y, index_y],
                                 self.c[index_y] - self.GI[index_y, index_y],
                                 self.c[index_y] + self.GI[index_y, index_y]])

        return x_coordinate, y_coordinate

    @classmethod
    def from_dict(cls, dict_var: dict):
        """
        Creates zonotope from dictionary

        Parameters
        ----------
        dict_var

        Returns
        -------

        """
        attr_list = ['center', 'GI', 'is_empty']
        input_list = parse_dict(dict_var, attr_list)
        is_empty_in = dict_var['is_empty']

        output_obj = cls(*input_list, is_empty=is_empty_in)
        return output_obj

    def __str__(self):
        """
        Returns information of zonotope as a string

        Returns
        -------

        """

        output_str = 'Zonotope\n'
        input_dict = super().to_info_string()
        for key in input_dict.keys():
            output_str += input_dict[key] + '\n'
        return output_str[:-2]

    def as_polynomial_zonotope(self):
        """
        Converts the zonotope to a polynomial zonotope (PZonotope)
        Returns
        -------

        """
        return PZonotope(super().get_center(), super().get_gi())

    def as_augmented_polynomial_zonotope(self):
        """
        Converts the zonotope to an augmented polynomial zonotope
        Returns
        -------

        """
        return AugPZonotope(super().get_center(), super().get_gi())

    def get_inf_norm(self) -> float:
        """
        Returns the infinity norm of the zonotope
        
        Returns
        -------

        """

        inf_norm = np.max(np.abs(self.c) + np.sum(np.abs(self.GI), axis=1).reshape((-1, 1)))
        return inf_norm

    def get_poly_contain(self, color="green", doi=None):
        """
        Returns the pypolycontain zonotope object in the plane of interest

        Parameters
        ----------
        color : str
        doi : list

        Returns
        -------

        """
        if doi is None:
            doi = [0, 1]

        return pp.zonotope(x=self.c[doi, 0], G=self.GI[doi, :], color=color)

    @staticmethod
    def over_approximate_2zonotope(z1_approximated, z2_approximated, z_out=None):
        """
        Overapproximate the two zonotopes with another zonotope
        In the case there is not an output variable specified a new Zonotope object is created

        Technique of the overapproximation
        https://github.com/JuliaReach/LazySets.jl/issues/229

        Parameters
        ----------
        z1_approximated
        z2_approximated
        z_out

        Returns
        -------

        """

        if z_out is None:
            z2_reduced = z2_approximated.to_order1()
            zonotope_out = Zonotope((z1_approximated.c + z2_reduced.c) / 2,
                                    np.concatenate((z1_approximated.c - z2_reduced.c, z1_approximated.GI +
                                                    z2_reduced.GI, z1_approximated.GI - z2_reduced.GI),
                                                   axis=1) / 2)
            return zonotope_out
        else:
            z_out.c[:] = (z1_approximated.c + z2_approximated.c) / 2
            np.concatenate((z1_approximated.c - z2_approximated.c, z1_approximated.GI +
                            z2_approximated.GI, z1_approximated.GI - z2_approximated.GI),
                           axis=1,
                           out=z_out.GI[:, :(1 + z1_approximated.GI.shape[1] + z2_approximated.shape[1])])

            z_out.GI /= 2

    def plot(self) -> None:
        """
        Plot the zonotope in xy plane

        Returns
        -------

        """
        zonotope_plot_obj = pp.zonotope(x=self.c[:2, 0], G=self.GI[:2, :])
        pp.visualize([zonotope_plot_obj], title=r'Zonotope')

    def to_order1(self, method='box', arg=-1):
        """
        Order reduction of zonotope. Resulting zonotope is of order 1.
        This method contains the following order reduction method
        -BOX
        -PCA
        -transform method (custom)
        -ExSe_y

        Parameters
        ----------
        method : str
            name of method
        arg : Any

        Returns
        -------

        Zonotope of order 1
        """

        reduced_zonotope = None
        if method == 'box':
            reduced_zonotope = Zonotope.order_reduction_box_method(self)
        elif method == 'pca':
            reduced_zonotope, _ = Zonotope.order_reduction_pca(self)
        elif method == 'transform':
            reduced_zonotope, _ = Zonotope.order_reduction_transform_method(self, arg)
            pass
        elif method == 'transform2':
            reduced_zonotope = Zonotope.overapproximation_ex_se_y(self, arg)
        return reduced_zonotope

    def get_volume(self) -> float:
        """
        Get the volume of the zonotope
        Returns
        -------

        """
        if self.get_order() == 1:
            return np.abs(np.linalg.det(self.GI))
        else:
            dominant_gen = get_dominant_generators(self.GI)
            return np.sqrt(np.linalg.det(np.dot(dominant_gen, dominant_gen.T)))

    @staticmethod
    def overapproximation_ex_se_y(input_zonotope, combinations):
        """
        Reduces the order of zonotope using ExSe_y method

        Parameters
        ----------
        input_zonotope
        combinations

        Returns
        -------

        """

        norm = np.linalg.norm(input_zonotope.GI, axis=0)

        # Sort the norm of each generator from high to low
        norm_sorted = np.flip(np.argsort(norm))

        # shuffle matrix
        input_zonotope.GI[:, :] = input_zonotope.GI[:, norm_sorted]

        if combinations is None:
            n = input_zonotope.GI[:, :].shape[0]
            transform_mat = copy.deepcopy(input_zonotope.GI[:, :n])
            return Zonotope.apply_order_reduction(input_zonotope, transform_mat)
        else:
            temp_mat = np.empty(input_zonotope.GI.shape, dtype=np.float)
            current_volume = -1
            current_zonotope = Zonotope
            for comb_i in combinations:
                transform_mat = copy.deepcopy(input_zonotope.GI[:, comb_i])
                try:
                    zono_i = Zonotope.apply_order_reduction(input_zonotope, transform_mat, temp_mat)
                    volume = zono_i.get_volume()
                    if volume < current_volume or current_volume == -1:
                        current_volume = volume
                        current_zonotope = zono_i
                except Exception as e:
                    print(e)
                    pass

            return current_zonotope

    @staticmethod
    def order_reduction_transform_method(input_zonotope, y=-1):
        """

        Parameters
        ----------
        input_zonotope
        y

        Returns
        -------

        """

        if y == -1:
            norm = np.linalg.norm(input_zonotope.GI, axis=0)

            # Sort the norm of each generator from high to low
            norm_sorted = np.flip(np.argsort(norm))

            # shuffle matrix
            input_zonotope.GI[:, :] = input_zonotope.GI[:, norm_sorted]

            # Get transform matrix
            n = len(input_zonotope.get_center())
            transform_matrix = np.zeros((n, n))
            m = 0
            for i in range(input_zonotope.GI.shape[1]):
                is_valid = True
                for j in range(m):
                    # Checks if two vectors do not align with each other
                    # Assumption: there are no 0 vectors. Therefore
                    # If vectors align than the result would be 1

                    if (abs(input_zonotope.GI[:, i].flatten().dot(transform_matrix[:, j])) /
                            (np.linalg.norm(input_zonotope.GI[:, i]) * np.linalg.norm(transform_matrix[:, j])) > 0.975):
                        is_valid = False
                        break
                if is_valid:
                    transform_matrix[:, m] = input_zonotope.GI[:, i]
                    m += 1
                if m == n:
                    break

            # If there is no valid transform matrix than use default box method
            if np.abs(np.linalg.det(transform_matrix)) < 0.001:
                reduced_zonotope = Zonotope.order_reduction_pca(input_zonotope)
                return reduced_zonotope

            np.dot(np.linalg.inv(transform_matrix), input_zonotope.GI, input_zonotope.GI)

            # Get basis in transformed frame
            new_generators = np.diag(np.sum(np.abs(input_zonotope.GI), axis=1))

            # Transform it back
            new_gi = np.dot(transform_matrix, new_generators)

            return Zonotope(input_zonotope.c, new_gi)
        return None, None

    @staticmethod
    def apply_order_reduction(input_zonotope, transform_matrix: np.ndarray, temp_matrix=None):
        """
        Applies the order reduction

        Parameters
        ----------
        input_zonotope
        transform_matrix
        temp_matrix

        Returns
        -------

        """

        # Transform zonotope with transform matrix that the
        if temp_matrix is None:
            np.dot(np.linalg.inv(transform_matrix), input_zonotope.GI, input_zonotope.GI)
            new_generators = np.diag(np.sum(np.abs(input_zonotope.GI), axis=1))
        else:
            np.dot(np.linalg.inv(transform_matrix), input_zonotope.GI, temp_matrix)
            new_generators = np.diag(np.sum(np.abs(temp_matrix), axis=1))

        # Transform it back
        new_gi = np.dot(transform_matrix, new_generators)

        return Zonotope(input_zonotope.c, new_gi)

    @staticmethod
    def order_reduction_pca(input_zonotope, threshold=-1):
        """

        Parameters
        ----------
        input_zonotope
        threshold

        Returns
        -------

        """
        if threshold == -1:
            generator_diff = np.concatenate((input_zonotope.GI, -input_zonotope.GI), axis=1)
        else:
            # TODO add threshold
            generator_diff = np.concatenate((input_zonotope.GI, -input_zonotope.GI), axis=1)
            pass
        c0 = np.dot(generator_diff, generator_diff.T)
        u, _, _ = np.linalg.svd(c0)

        # TODO add is_orthogonal
        return Zonotope.apply_order_reduction(input_zonotope, u)

    @staticmethod
    def order_reduction_box_method(input_zonotope):
        """
        overapproximates the high order zonotope with an interval.

        Parameters
        ----------
        input_zonotope

        Returns
        -------

        """
        gi_new = np.diag(np.sum(np.abs(input_zonotope.GI), axis=1))
        return Zonotope(input_zonotope.c, gi_new)

    def to_dict(self) -> dict:
        """

        Returns
        -------

        """

        return self.combine_dict(['is_empty', 'c', 'GI'])

    def transform(self, map_matrix: np.ndarray) -> None:
        """
        Transforms the zonotope
        Parameters
        ----------
        map_matrix

        Returns
        -------

        """
        self.c = map_matrix.dot(self.c)
        self.GI = map_matrix.dot(self.GI)

    def intersect_zonotope(self, input_zonotope) -> bool:
        """
        Checks whether two zonotopes intersect with each other

        Parameters
        ----------
        input_zonotope

        Returns
        -------

        """
        g_diff = np.concatenate((self.GI, -input_zonotope.GI), axis=1)
        center_diff = input_zonotope.c - self.c
        res = linprog(np.ones(g_diff.shape[1]), None, None, g_diff, center_diff, (-1, 1), method='revised simplex')

        return res['success']

    def is_in_domain(self, domain_obj: Type[Domain]) -> bool:
        """
        Check whether zonotope is in the Domain object.

        Parameters
        ----------
        domain_obj

        Returns
        -------

        """
        # translate the system
        center_diff = self.c - domain_obj.center
        abs_dist = np.sum(np.abs(self.GI), axis=1).reshape((-1, 1)) + np.abs(center_diff)

        # If one axis is longer than domain than the zonotope is not within the domain
        return not np.any(abs_dist > domain_obj.axis_length)


def plot_zonotope(zonotope_in: Zonotope) -> None:
    """
    In this function a 2D Zonotope is plotted.
    The chosen dimensions of the Zonotope are the first two entries.

    Parameters
    ----------
    zonotope_in

    Returns
    -------

    """

    zono_i = pp.zonotope(x=zonotope_in.c.flatten()[:2], G=zonotope_in.GI[:2, :], color='cyan')
    vert = pp.conversions.zonotope_to_V(zono_i)
    plt.plot(np.append(vert[:, 0], vert[0, 0]), np.append(vert[:, 1], vert[0, 1]), color='black')
    plt.fill(np.append(vert[:, 0], vert[0, 0]), np.append(vert[:, 1], vert[0, 1]), color='cyan', alpha=0.3)


def plot_trajectory(zonotope_list: List[Zonotope]) -> None:
    """
    Plot a list of zonotopes

    Parameters
    ----------
    zonotope_list

    Returns
    -------

    """
    plt.figure()

    for i in range(len(zonotope_list)):
        # zono_i = zonotope_list[i].get_poly_contain()
        plot_zonotope(zonotope_list[i])

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    return None


class PZonotope(BaseOperations):
    """
        PZonotope
    """

    @staticmethod
    def __add__(self, zonotope_in: Zonotope):
        """

        Parameters
        ----------
        self
        zonotope_in

        Returns
        -------

        """

        # super().__add__(zonotope_in)
        # # Because the generators of G are always dense the structure is predictable
        # if zonotope_in.get_g() is not None:
        #     if self.g_mat is None:
        #         n1 = self.g_mat.shape[1]
        #         n2 = zonotope_in.g_mat.shape[1]
        #         if n1 < n2:
        #             g_temp = zonotope_in.get_g()
        #             g_temp[:, :n1] += self.g_mat
        #         else:
        #             self.g_mat[:, :n2] += zonotope_in.g_mat
        #     else:
        #         self.g_mat = zonotope_in.g_mat
        # return self
        raise NotImplementedError

    dependent_generators = None

    def __init__(self, c: np.ndarray, gi: Union[np.ndarray, None], dependent_generators=None, e_mat=None, **kwargs):
        """

        Parameters
        ----------
        c
        gi
        dependent_generators
        e_mat
        kwargs
        """

        super().__init__(c, gi, **kwargs)

        if dependent_generators is not None:
            self.g_mat = np.concatenate((c, dependent_generators), axis=1)
            self.dependent_generators = self.g_mat[:, 1:]
            self.E = np.concatenate((np.zeros((e_mat.shape[0], 1)), e_mat), axis=1)
            self.c = self.g_mat[:, 0]
        else:
            self.g_mat = c
            self.E = np.zeros(c.shape)
        self.calculate_order()

    def __matmul__(self, transform_matrix: np.ndarray):
        """

        Parameters
        ----------
        transform_matrix

        Returns
        -------

        """
        super().__matmul__(transform_matrix)
        self.g_mat.dot(transform_matrix)

    def __str__(self) -> str:
        """

        Returns
        -------

        """
        input_dict = super().to_info_string()
        output_str = 'Polynomial Zonotope\n'
        base_attr_list = ['g_mat', 'E', 'compress_array']

        for attr in base_attr_list:
            value = getattr(self, attr)

            if type(value) == np.ndarray:
                value = value.shape

            input_dict[attr] = '%s: %s' % (attr.ljust(8), str(value) if value is not None else 'not defined')

        key_list = ['c', 'g_mat', 'GI', 'E', 'exponent', 'n_low', 'compress_array']

        for key in key_list:
            output_str += input_dict[key] + '\n'

        return output_str[:-2]

    def empty_gi(self) -> None:
        """

        Returns
        -------

        """
        if self.GI is None:
            pass

        elif self.get_g() is not None:
            self.g_mat = np.concatenate((self.g_mat, self.GI), axis=1)

            if not (np.array_equal(self.E, np.zeros(self.E.shape))):
                self.E = block_diag(self.E, np.fliplr(np.eye(self.GI.shape[1])))

            else:
                self.E = np.concatenate((self.E, np.fliplr(np.eye(self.GI.shape[1]))), axis=1)

        else:
            # IS NOT GOOD UNSAFE
            self.g_mat = self.GI
            self.E = np.fliplr(np.eye(self.g_mat.shape[1]))
        self.GI = None

    def get_center(self) -> np.ndarray:
        """
        Get center of the set

        Returns
        -------

        """
        # Returns the center
        return self.g_mat[:, 0]

    # @timeit
    def get_e(self) -> np.ndarray:
        """
        Returns the matrix which represents the exponents of each factor for each monomial

        Returns
        -------

        """
        return self.E

    def get_g(self) -> np.ndarray:
        """
        Returns the matrix of which represents the dependent generators and the center

        Returns
        -------

        """

        return self.g_mat

    def get_generators(self) -> np.ndarray:
        """
        Returns the generators of the set

        Returns
        -------

        """
        if self.g_mat.shape[1] > 1:
            return self.g_mat[:, 1:]

        else:
            print('There are not any dependent generators')
            raise Exception

    def get_order(self) -> float:
        """
        Get the order of the set (number of generators / dimension)
        Returns
        -------

        """
        n_generators = 0
        if self.get_g() is not None:
            n_generators += self.g_mat.shape[1]

        if self.get_gi() is not None:
            n_generators += self.GI.shape[1]

        return n_generators / super().get_set_dimension()

    def minkowski_zonotope_addition(self, polynomial_zonotope_in):
        """
        Add the polynomial zonotope to the set

        Parameters
        ----------
        polynomial_zonotope_in

        Returns
        -------

        """

        # This function assumes that coefficients gamma are different
        super().minkowski_zonotope_addition(polynomial_zonotope_in)

        if polynomial_zonotope_in.get_g() is not None:

            if self.get_g() is not None:
                self.g_mat = np.concatenate((self.get_g(), polynomial_zonotope_in.get_g()), axis=1)

            else:
                self.g_mat = polynomial_zonotope_in.get_g()

    @staticmethod
    def over_approximate_2polynomial_zonotope(pz1, pz2):
        """
        Overapproximates two polynomial zonotopes with 1 zonotope

        Parameters
        ----------
        pz1
        pz2

        Returns
        -------

        """

        z1_overapproximated = pz1.to_zonotope()
        z2_overapproximated = pz2.to_zonotope()
        return z1_overapproximated, z2_overapproximated

    def to_zonotope(self, out_zonotope=None, same_size=True):
        """
        Creates Zonotope from Polynomial Zonotope

        Parameters
        ----------
        out_zonotope
        same_size

        Returns
        -------

        """

        # Count all vectors without factors. However this is not the case for these classes

        # get all generators with a convex monomial
        h_1 = np.where(np.sum(np.remainder(self.E, 2), axis=0) == 0)[0]

        # Get all generators with a non-convex monomial
        k_1 = np.delete(np.arange(0, self.E.shape[1]), h_1)

        if out_zonotope is None:
            new_center = np.sum(self.c + 1 / 2 * np.sum(self.g_mat[:, h_1], axis=1))
            if self.GI is not None:
                new_gi = np.block([0.5 * self.g_mat[:, h_1], self.g_mat[:, k_1], self.GI])
            else:
                new_gi = np.block([0.5 * self.g_mat[:, h_1], self.g_mat[:, k_1]])

            return Zonotope(new_center, new_gi)

        else:
            np.sum(self.c + 1 / 2 * np.sum(self.g_mat[:, h_1], axis=1), out_zonotope.c)
            if self.GI is not None:
                if same_size:
                    np.concatenate((0.5 * self.g_mat[:, h_1], self.g_mat[:, k_1], self.GI), axis=1, out=out_zonotope.GI)
                else:
                    np.concatenate((0.5 * self.g_mat[:, h_1], self.g_mat[:, k_1], self.GI), axis=1,
                                   out=out_zonotope.GI[:, :(h_1.size + k_1.size + self.GI.shape[1])])
            else:
                if same_size:
                    np.concatenate((0.5 * self.g_mat[:, h_1], self.g_mat[:, k_1]), axis=1, out=out_zonotope.GI)
                else:
                    np.concatenate((0.5 * self.g_mat[:, h_1], self.g_mat[:, k_1], self.GI), axis=1,
                                   out=out_zonotope.GI[:, :(h_1.size + k_1.size)])
            return out_zonotope

    def get_json_dict(self, attr_oi: str) -> dict:
        """
        Creates a dictionary of the attributes of the object

        Parameters
        ----------
        attr_oi

        Returns
        -------

        """
        try:
            variable_of_interest = getattr(self, attr_oi)
        except Exception as e:
            print(e)
            print('retrieving attribute has failed')
            raise Exception

        if attr_oi in ['G', 'dependent_generators']:
            return matrix_handler(attr_oi, variable_of_interest, self.is_empty)
        elif attr_oi in ['E']:
            return ndarray_to_list('E', self.get_e())

        return super().get_json_dict(attr_oi)

    def calculate_order(self) -> float:
        """
        Calculates the order of the set
        Returns
        -------

        """
        n_dependent = 0
        n_independent = 0
        if hasattr(self, 'g_mat'):
            if getattr(self, 'g_mat') is not None:
                n_dependent = self.g_mat.shape[1]
        if hasattr(self, 'GI'):
            if getattr(self, 'GI') is not None:
                n_independent = self.GI.shape[1]

        order = (n_dependent + n_independent) / len(self.c)
        return order


class AugPZonotope(PZonotope):
    """
    AugPZonotope
    """

    def __add__(self, zonotope_in: Zonotope):
        """

        Parameters
        ----------
        zonotope_in

        Returns
        -------

        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dict_var: dict):
        """
        Creates AugPZonotope from dictionary

        Parameters
        ----------
        dict_var

        Returns
        -------

        """
        attr_list = ['center', 'GI', 'dependent_generators', 'E', 'tail', 'head', 'compress_array', 'is_empty']
        input_list = parse_dict(dict_var, attr_list)

        is_empty_in = dict_var['is_empty']

        output_obj = cls(*input_list, is_empty=is_empty_in)
        if 'dimension_low' in dict_var.keys():
            output_obj.set_dim_low(dict_var['dimension_low'])

        output_obj.set_exponent(dict_var['exponent'])
        return output_obj

    @classmethod
    def from_json(cls, json_dict: str):
        """
        Creates object from json string

        Parameters
        ----------
        json_dict

        Returns
        -------

        """
        dict_var = json.loads(json_dict)
        return AugPZonotope.from_dict(dict_var)

    def to_dict(self) -> dict:
        """
        Converts AugPZonotope to dictionary

        Returns
        -------

        """
        return self.combine_dict(['c', 'GI', 'dependent_generators',
                                  'E', 'tail', 'head',
                                  'tail_filtered_dict', 'compress_array', 'is_empty',
                                  'exponent', 'dimension_low'])

    def to_json(self) -> str:
        """
        Converts AugPZonotope to json string
        Returns
        -------

        """
        output_dict = self.to_dict()
        return json.dumps(output_dict)

    def __init__(self, c: np.ndarray, gi, dependent_generators=None, e_mat=None, tail=None, head=None,
                 compress_array=None,
                 **kwargs):
        """

        Parameters
        ----------
        c
        gi
        dependent_generators
        e_mat
        tail
        head
        compress_array
        kwargs
        """

        if dependent_generators is not None and e_mat is not None:
            if dependent_generators.shape[1] != e_mat.shape[1]:
                print('Mismatch ')
                raise Exception
        super().__init__(c, gi, dependent_generators, e_mat, **kwargs)
        self.tail = tail
        self.head = head

        self.compress_array = compress_array
        self.head_dict = {}
        self.tail_dict = {}
        self.tail_filtered_dict = {}
        self.even_gen = np.empty((1, 3))
        self.odd_gen = np.empty((1, 3))

        if tail is not None:
            if head is not None:
                self.__init_dict()
        else:
            self.__init_dict_empty()

        if 'reorder_array' in kwargs.keys():
            self.reorder_array = kwargs['reorder_array']

        self.set_odd_even_generators()

    def get_even_gen(self) -> np.ndarray:
        """
        Get the indices of the dependent generators with
        Returns
        -------

        """
        return self.even_gen

    def get_odd_gen(self) -> np.ndarray:
        """

        Returns
        -------

        """
        return self.odd_gen

    def set_odd_even_generators(self) -> None:
        """

        Returns
        -------

        """
        ones_mat = np.ones(self.E.shape)
        e_modulo = np.remainder(self.E, 2)
        e_diff = ones_mat - e_modulo
        e_prod = np.prod(e_diff, axis=0)

        h_t = np.where(e_prod == 1)[0]

        # Set all even except center
        self.even_gen = h_t[1:]

        # Remove all even indices
        self.odd_gen = np.delete(np.arange(0, self.E.shape[1]), h_t)

    def __init_dict(self) -> None:
        """
        Creates dictionary related to the first index and last index of each monomial

        Returns
        -------

        """

        n_max = int(np.max(np.append(self.tail, self.head)) + 1)

        # Find out what index is in front and what is in the back
        for i in range(0, n_max):
            self.head_dict[i] = np.where(self.head == i)[0]
            self.tail_dict[i] = np.where(self.tail == i)[0]

        # Create tail filtered dict
        for i in range(n_max - 1, -1, -1):
            if i == n_max - 1:
                self.tail_filtered_dict[i] = self.tail_dict[i].flatten()

            else:
                self.tail_filtered_dict[i] = np.concatenate((self.tail_dict[i],
                                                             self.tail_filtered_dict[i + 1])).flatten()

    @classmethod
    def create_empty_polynomial_zonotope(cls, spz_1, spz_2, n_low: int,
                                         new_e_matrix: np.ndarray, compress_array: np.ndarray):
        """
        Creates empty Polynomial zonotope objects in order to allocate memory for the monomial transformer

        Parameters
        ----------
        spz_1 : AugPZonotope
            First input polynomial zonotope, which is on the left side of the kronecker product
        spz_2 : AugPZonotope
            Second input polynomial zonotope, which is on the right side of the kronecker product
        n_low : int
            Dimension of the differential equation
        new_e_matrix : ndarray
            E matrix for the empty object
        compress_array : ndarray
            The inverse array required for compression the generators

        Returns
        -------
        Empty Polynomial zonotope object with memory allocated for generators
        """

        # Get monomial order of output
        new_exponent = spz_1.get_exponent() + spz_2.get_exponent()

        # Amount of generators of the polynomial zonotope
        n_gen = comb(new_exponent + n_low, new_exponent) - 1

        # Unique monomials
        n_rows = AugPZonotope.__get_n_rows(spz_1, spz_2, n_low)

        # index of front and back
        heads_array, tails_array, reorder_array = AugPZonotope.create_tail_and_heads(spz_1, spz_2, n_low, n_rows)

        new_center = np.empty((n_rows, 1), dtype=np.float64)
        new_g = np.empty((n_rows, n_gen), dtype=np.float64)

        # create new SPZ object
        spz_new = AugPZonotope(new_center, None,
                               new_g, new_e_matrix,
                               tails_array,
                               heads_array,
                               compress_array, is_empty=True, reorder_array=reorder_array, exponent=new_exponent)
        return spz_new

    @staticmethod
    def __get_n_rows(spz_1, spz_2, n_low: int):
        """
        Get dimension of resulting polynomial zonotope after applying the reduced kronecker product of
        spz_1 and spz_2

        Parameters
        ----------
        spz_1 : AugPZonotope
            Left side polynomial zonotope
        spz_2 : AugPZonotope
            Right side polynomial zonotope
        n_low : int
            Dimension of differential equation

        Returns
        -------
        Amount of rows of the output polynomial zonotope
        """

        tail1 = spz_1.get_filtered_tail()
        head2 = spz_2.get_head()
        n_rows = 0
        for i in np.arange(n_low):
            # Get indices to use of G
            tail1_index = tail1[i]
            head2_index = head2[i]

            # Extra rows memory location
            n_rows += (len(tail1_index) * len(head2_index))
        return n_rows

    @staticmethod
    def create_tail_and_heads(spz_1, spz_2, n_low: int, n_rows: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create the new list of the heads and tails of the new polynomial zonotope of the kronecker product

        Parameters
        ----------
        spz_1 : AugPZonotope
        spz_2 : AugPZonotope
        n_low : int
        n_rows : int

        Returns
        -------

        """

        tail1 = spz_1.get_filtered_tail()
        head2 = spz_2.get_head()
        heads_array = np.zeros(n_rows)
        tails_array = np.zeros(n_rows)
        reorder_array = np.zeros(n_rows, dtype=np.int64)
        start_index_head = 0

        for i in range(0, n_low):
            n_new = head2[i].size * tail1[i].size
            head_i = np.kron(spz_1.head[tail1[i]], np.ones((1, head2[i].size))).reshape((1, -1))
            tail_i = np.repeat(spz_2.tail[head2[i]], tail1[i].size)
            heads_array[start_index_head:(start_index_head + n_new)] = head_i
            tails_array[start_index_head:(start_index_head + n_new)] = tail_i
            start_index_head += head_i.size

        start_index_reorder = 0
        for i in range(n_low):
            ii_indices = np.argwhere(tails_array == i).flatten()
            reorder_array[ii_indices] = np.arange(start_index_reorder, start_index_reorder + ii_indices.size)
            start_index_reorder += ii_indices.size
        idx = np.empty_like(reorder_array)
        idx[reorder_array] = np.arange(len(reorder_array))
        heads_array_new = heads_array[idx]
        tails_array_new = tails_array[idx]
        return heads_array_new, tails_array_new, idx

    def __init_dict_empty(self) -> None:
        """
        Create the heads and tails for the Augmented Polynomial zonotope

        Returns
        -------

        """
        self.head = np.arange(self.get_set_dimension())
        self.tail = np.arange(self.get_set_dimension())
        self.__init_dict()

    def get_compress_array(self) -> np.ndarray:
        """
        Get the compress array, which is used for the compact operation
        Returns
        -------

        """
        return self.compress_array

    def get_tail(self) -> dict:
        """
        Get index of last x_k of the sequence of x_ix_jx_k

        Returns

        -------

        """

        return self.tail_dict

    def get_head(self) -> dict:
        """
        Get index of first x_i of the sequence of x_ix_jx_k

        Returns
        -------

        """

        return self.head_dict

    def get_filtered_tail(self) -> dict:
        """
        Get dictionary of a list of rows which end on a monomial which has a lower value than i

        Returns
        -------

        """
        return self.tail_filtered_dict

    def get_over_approx_inf(self) -> float:
        """
        Get the infinity norm of the zonotope which overapproximates the polynomial zonotope.
        Returns
        -------

        """
        row_sum = self.get_center() + 0.5 * np.sum(self.g_mat[:, ], axis=1)
        if self.even_gen.size != 0:
            row_sum += 0.5 * np.sum(np.abs(self.g_mat[:, self.even_gen]), axis=1)
        if self.odd_gen.size != 0:
            row_sum += np.sum(np.abs(self.g_mat[:, self.odd_gen]), axis=1)

        return np.max(row_sum)

    def to_zonotope(self, out_zonotope=None, same_size=True) -> Zonotope:
        """
        Creates Zonotope from Augmented polynomial zonotope

        Parameters
        ----------
        out_zonotope
        same_size

        Returns
        -------

        """

        # Count all vectors without factors. However this is not the case for these classes

        # TODO SPAGHETTI
        # get all generators with a convex monomial
        if out_zonotope is None:
            new_center = np.sum(self.c + 1 / 2 * np.sum(self.g_mat[:, self.even_gen], axis=1))
            if self.GI is not None:
                new_gi = np.block([0.5 * self.g_mat[:, self.even_gen], self.g_mat[:, self.odd_gen], self.GI])
            else:
                new_gi = np.block([0.5 * self.g_mat[:, self.even_gen], self.g_mat[:, self.odd_gen]])

            return Zonotope(new_center, new_gi)

        else:
            out_zonotope.c[:] = self.c.reshape((-1, 1)) + 1 / 2 * np.sum(self.g_mat[:, self.even_gen], axis=1,
                                                                         keepdims=True)
            if self.GI is not None:
                if same_size:
                    np.concatenate((0.5 * self.g_mat[:, self.even_gen], self.g_mat[:, self.odd_gen], self.GI), axis=1,
                                   out=out_zonotope.GI)
                else:
                    np.concatenate((0.5 * self.g_mat[:, self.even_gen], self.g_mat[:, self.odd_gen], self.GI), axis=1,
                                   out=out_zonotope.GI[:, :(self.even_gen.size + self.odd_gen.size + self.GI.shape[1])])
            else:
                if same_size:
                    np.concatenate((0.5 * self.g_mat[:, self.even_gen], self.g_mat[:, self.odd_gen]), axis=1,
                                   out=out_zonotope.GI)
                else:
                    np.concatenate((0.5 * self.g_mat[:, self.even_gen], self.g_mat[:, self.odd_gen]), axis=1,
                                   out=out_zonotope.GI[:, :(self.even_gen.size + self.odd_gen.size)])
            return out_zonotope

    def get_json_dict(self, attr_oi: str) -> dict:
        """
        Get dictionary with json friendly variable
        Parameters
        ----------
        attr_oi

        Returns
        -------

        """
        try:
            variable_of_interest = getattr(self, attr_oi)
        except Exception as e:
            print(e)

            print('retrieving %s attribute has failed' % attr_oi)
            raise Exception

        if attr_oi in ['head_filtered_dict', 'tail_filtered_dict']:
            return get_dict_json(attr_oi, variable_of_interest)
        elif attr_oi in ['head', 'tail']:
            return ndarray_to_list(attr_oi, variable_of_interest)
        elif attr_oi in ['compress_array']:
            return ndarray_to_list(attr_oi, variable_of_interest)

        return super().get_json_dict(attr_oi)


@nb.njit(fastmath=True, parallel=False)
def kronecker_product(a_in: np.ndarray, b_in: np.ndarray, out: np.ndarray):
    """
    Kronecker product of two matrices.
    Parameters
    ----------
    a_in : np.ndarray
    left matrix

    b_in : np.ndarray
    right matrix

    out : np.ndarray
    output matrix

    Returns
    -------

    """
    # Kronecker product
    n2 = b_in.shape[0]
    m2 = b_in.shape[1]
    for i1 in nb.prange(a_in.shape[0]):
        for i2 in nb.prange(b_in.shape[0]):
            for j1 in nb.prange(a_in.shape[1]):
                for j2 in nb.prange(b_in.shape[1]):
                    out[n2 * i1 + i2, m2 * j1 + j2] = a_in[i1, j1] * b_in[i2, j2]

    return out


def get_dict_json(attr_name: str, possible_dict: dict) -> dict:
    """
    returns a {key : item}. If the item is a np.ndarray then it is converted to a list

    Parameters
    ----------
    attr_name
    possible_dict

    Returns
    -------

    """
    if type(possible_dict[0]) == np.ndarray:
        output_tail = dict_ndarray_to_list(possible_dict)
        return {attr_name: output_tail}
    else:
        return {attr_name: possible_dict}


def dict_ndarray_to_list(input_dict: dict) -> dict:
    """
    Convert a dictionary with a np.ndarray to a dictionary with a list

    Parameters
    ----------
    input_dict

    Returns
    -------

    """
    # TODO rename function
    output_dict = copy.deepcopy(input_dict)
    for key in output_dict.keys():
        if type(output_dict[key]) == np.ndarray:
            output_dict[key] = output_dict[key].tolist()
    return output_dict


def dict_list_to_ndarray(input_dict: dict) -> dict:
    """
    Converts a dictionary with a list to a dictionary with a np.ndarray

    Parameters
    ----------
    input_dict

    Returns
    -------

    """
    # TODO rename function
    output_dict = copy.deepcopy(input_dict)
    for key in output_dict.keys():
        if type(output_dict[key]) == list:
            output_dict[key] = np.array(output_dict[key])
    return output_dict


def ndarray_to_list(name: str, input_obj: np.ndarray):
    """
    Converts a dictionary with np.ndarray to a dictionary with a list
    TODO double?
    Parameters
    ----------
    name
    input_obj

    Returns
    -------

    """
    # TODO rename function
    if type(input_obj) == np.ndarray:
        return {name: input_obj.tolist()}
    else:
        return {name: input_obj}


def dict_to_ndarray(input_dict: dict, rows: int) -> np.ndarray:
    """
    Assign the key number to the places of the output array

    Parameters
    ----------
    input_dict
    rows

    Returns
    -------

    """
    # TODO rename function
    output_array = np.empty(rows, dtype=np.int)
    for key in input_dict.keys():
        output_array[input_dict[key]] = key

    return output_array


def matrix_handler(name: str, input_obj, is_empty: bool):
    """
    converts np.ndarray to a dictionary with a json friendly variables.
    If is_empty = True then only the shape of the array is saved
    If false then the entire matrix is saved in the dictionary variable

    Parameters
    ----------
    name
    input_obj
    is_empty

    Returns
    -------

    """
    # TODO rename function
    if is_empty:
        if type(input_obj) == np.ndarray:
            return {name: input_obj.shape}
        else:
            return ndarray_to_list(name, input_obj)
    else:
        return ndarray_to_list(name, input_obj)


# @timeit
@nb.njit(fastmath=True, parallel=False)
def compress_generators(g_in: np.ndarray, x: np.ndarray, out: np.ndarray):
    """
    Sums the columns with the same indices in array x.
    This method is used for the compact operation

    Parameters
    ----------
    g_in
    x
    out

    Returns
    -------

    """
    for i in nb.prange(g_in.shape[0]):
        for j in np.arange(g_in.shape[1]):
            out[i, x[j]] += g_in[i, j]
    return out


def parse_dict(dict_var: dict, attr_list: List[str]) -> list:
    """
    Parses the dictionary with json friendly variables to
    information that can be used for the set classes.

    Parameters
    ----------
    dict_var
    attr_list

    Returns
    -------

    """
    # TODO rename function
    # Create list in correct order
    # input_list = [json_dict[attr_i] for attr_i in attr_list]
    input_dict = {}
    if 'is_empty' in dict_var.keys():
        is_empty_in = dict_var['is_empty']
    else:
        is_empty_in = False

    for attr_i in attr_list:
        el = dict_var[attr_i]
        if attr_i in ['center', 'GI', 'dependent_generators']:
            if dict_var[attr_i] is not None:
                if is_empty_in:
                    input_dict[attr_i] = np.empty(tuple(el), dtype=np.float64)
                else:
                    input_dict[attr_i] = np.array(el)

                if attr_i == 'center':
                    input_dict[attr_i] = input_dict[attr_i].reshape((-1, 1))
                pass
            else:
                input_dict[attr_i] = el
        elif attr_i in ['compress_array', 'E', 'tail', 'head']:
            input_dict[attr_i] = np.array(el)
            if attr_i == 'E':
                if not np.any(input_dict[attr_i][:, 0]):
                    input_dict[attr_i] = input_dict[attr_i][:, 1:]
            pass
        elif attr_i in ['tail_filtered_dict']:
            input_dict[attr_i] = dict_list_to_ndarray(el)
        else:
            print('%s was not caught' % attr_i)

    input_list = [input_dict[attr_i] for attr_i in input_dict.keys()]
    return input_list


def get_dominant_generators(generators: np.ndarray, threshold=10 ** -6) -> np.ndarray:
    """
    Returns the generators sorted based on length.
    This method does not return generators below a threshold 'threshold'

    Parameters
    ----------
    generators
    threshold

    Returns
    -------

    """
    norm = np.linalg.norm(generators, axis=0)
    n = np.sum(norm > threshold)

    # Sort the norm of each generator from high to low
    norm_sorted = np.flip(np.argsort(norm))

    # shuffle matrix
    generators[:, :] = generators[:, norm_sorted]

    return generators[:, :(generators.shape[1] + n)]


class IntervalHull(Zonotope):
    """
    This class represents an interval hul, which is a point symmetric set
    and the axis are parallel with the coordinates axes
    """

    def __init__(self, c: np.ndarray, gi: np.ndarray):
        super().__init__(c, gi)
        self.bounds = np.ndarray
        self.set_border()

    @staticmethod
    def overapproximate2interval_hull(ih1, ih2):
        """
        Approximates 2 interval hull with another interval hull

        Parameters
        ----------
        ih1 : IntervalHull
        ih2 : IntervalHull

        Returns
        -------

        """
        new_center = (ih1.c + ih2.c) / 2
        new_gi = np.diag(
            np.amax(np.concatenate((ih1.get_upper_bound(), ih2.get_upper_bound()), axis=1) - new_center, axis=1))

        return IntervalHull(new_center, new_gi)

    def set_border(self) -> None:
        """
        Set the bounds of the interval

        Returns
        -------

        """
        output = np.empty((self.get_set_dimension(), 2))
        output[:, 0] = self.c.flatten() - np.abs(np.sum(self.GI, axis=1))
        output[:, 1] = self.c.flatten() + np.abs(np.sum(self.GI, axis=1))
        self.bounds = output

    def get_lower_bound(self) -> np.ndarray:
        """
        Get the lower bound of the interval hull
        Returns
        -------

        """
        return self.bounds[:, 0].reshape((-1, 1))

    def get_upper_bound(self) -> np.ndarray:
        """
        Get the upper bound of the interval hull
        Returns
        -------

        """
        return self.bounds[:, 1].reshape((-1, 1))


class ZonotopeList:
    """
    ZonotopeList
    This class contains a list of zonotopes. The main purpose of this class is plotting

    """

    def __init__(self, zonotope_list: List[Zonotope], color='green', doi=None, name=None, is_interval=False, **kwargs):
        """
        Initializes ZonotopeList object

        Parameters
        ----------
        zonotope_list : List[Zonotope]
        color
        doi
        name
        """

        if doi is None:
            doi = [0, 1]
        self.name = name
        self.zonotope_list = zonotope_list
        self.color = color
        self.doi = doi
        self.vertex_list = []
        self.is_interval = is_interval

        # override
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.n_zonotopes = len(self.zonotope_list)

    def visualize(self, ax: plt.axes) -> None:
        """
        Visualizes a set of zonotopes

        Parameters
        ----------
        ax

        Returns
        -------

        """
        zonotope_list = [zono_i.get_poly_contain(self.color, self.doi) for zono_i in self.zonotope_list]
        pp.visualize(zonotope_list, ax=ax)

    def to_dict(self) -> dict:
        """
        Converts the ZonotopeList object to a dictionary

        Returns
        -------

        """

        output_dict = {'color': self.color}
        output_dict.update(self.get_vertices_dict())
        output_dict.update({'name': self.name})
        output_dict.update({'doi': self.doi})
        return output_dict

    def get_vertices_dict(self) -> dict:
        """
        Get list of vertices of the ZonotopeList in the xi xj plane

        Returns
        -------

        """
        output_x, output_y = self.get_vertex_list()
        return {'vertices_list': {'x': output_x.tolist(),
                                  'y': output_y.tolist()}
                }

    def get_vertex_list(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the vertices of the ZonotopeList

        Parameters
        ----------


        Returns
        -------

        """
        max_vert = 2 * self.zonotope_list[0].GI.shape[1]
        output_x = np.empty((len(self.zonotope_list), max_vert + 1))
        output_y = np.empty((len(self.zonotope_list), max_vert + 1))

        for i in range(len(self.zonotope_list)):
            pp_zono = self.zonotope_list[i].get_poly_contain(self.color, self.doi)
            vertices = pp.conversions.zonotope_to_V(pp_zono)

            if vertices.shape[0] == max_vert:
                output_x[i, :-1] = vertices[:, 0]
                output_x[i, -1] = vertices[0, 0]

                output_y[i, :-1] = vertices[:, 1]
                output_y[i, -1] = vertices[0, 1]
            else:
                output_x[i, :vertices.shape[0]] = vertices[:, 0]
                output_x[i, vertices.shape[0]:] = vertices[0, 0]

                output_y[i, :vertices.shape[0]] = vertices[:, 1]
                output_y[i, vertices.shape[0]:] = vertices[0, 1]

        return output_x, output_y

    def __len__(self):
        return self.n_zonotopes


class PZonotopeList:
    """
    class object which represents a list of polynomial zonotopes.
    main purpose is lifting mapping and projecting
    """

    def get_dim_low(self) -> int:
        """
        Returns the dimension of the lower dimension R^n

        Returns
        -------

        """
        return self.dimension_low

    def get_dim_low_json(self) -> dict:
        """
        Returns the lower dimensional space as dictionary variable
        Returns
        -------

        """
        return {'dimension_low': self.get_dim_low()}

    def get_max_order(self) -> int:
        """
        Returns the highest order of the monomial in the lifted space

        Returns
        -------

        """
        return self.MaxOrder

    def get_max_order_json(self) -> dict:
        """
        Get the maximum monomial order of the Polynomial zonotope list

        Returns
        -------

        """
        return {'max_order': self.get_max_order()}

    @classmethod
    def from_json(cls, input_json: str):
        """
        Create PZList from input string which is in json format.

        Parameters
        ----------
        input_json : str
        Information string containing all information of polynomial zonotope list in json format

        Returns
        -------

        """
        input_dict = json.loads(input_json)
        return PZonotopeList.from_dict(input_dict)

    @classmethod
    def from_dict(cls, input_dict: dict):
        """
        Creates the a Polynomial zonotope list based on a dictionary variable

        Parameters
        ----------
        input_dict : dict
            variable containing the necessary keywords to construct the Polynomial zonotope list
        Returns
        -------
        PolynomialZonotopeList object
        """
        pz_list_in = [AugPZonotope.from_dict(pz_i) for pz_i in input_dict['pz_list']]
        return cls(pz_list_in, input_dict['dimension_low'], input_dict['max_order'])

    def to_dict(self) -> dict:
        """
        Converts the Polynomial zonotope list to a dictionary
        Returns
        -------

        """
        output_dict = {'pz_list': [pz_i.to_dict() for pz_i in self.polynomial_zonotope_list]}
        output_dict.update(self.get_dim_low_json())
        output_dict.update(self.get_max_order_json())

        return output_dict

    def to_json(self):
        """
        Converts the Polynomial zonotope list to a json string
        Returns
        -------

        """
        output_dict = self.to_dict()

        return json.dumps(output_dict)

    @staticmethod
    def _create_polynomial_zonotope_template(n_low: int, max_order: int, monomials_used=None) -> List[AugPZonotope]:
        """
        Allocate memory for the lifted state.
        This is done by creating empty Augmented Polynomial zonotope objects

        Parameters
        ----------
        n_low : int
            Dimension of differential equation
        max_order : int
            Amount of Lie derivative that have to be computed for lifting
        monomials_used : List[tuple]
            Monomials that are used (Not implemented)

        Returns
        -------
        List of empty polynomial zonotopes
        """

        spz_base = AugPZonotope(np.empty((n_low, 1), dtype=np.float64),
                                np.empty((n_low, n_low), dtype=np.float64), is_empty=True, exponent=1)
        spz_base.empty_gi()

        # create matrices for E
        e_list, inverse_list = PZonotopeList._create_e2(n_low, max_order)

        if monomials_used is None:

            # Amount of leading polytopes. This is equal to the highest 2^x < max order
            max_shifts = int(np.floor(np.log2(max_order)) + 1)

            # Set the initial set (zonotope) as the current polytope
            current_polytope = spz_base
            polytope_queue_all = []

            # get dimension of the differential equation in the lower space
            n_low = spz_base.get_set_dimension()

            for i in range(0, max_shifts):

                # Update queue with all previous polytopes
                polytope_queue = copy.deepcopy(polytope_queue_all)
                while polytope_queue:

                    # Remove first element from queue
                    poly_i = polytope_queue[0]
                    polytope_queue.pop(0)

                    result_order = current_polytope.get_exponent() + poly_i.get_exponent()
                    # Determine next order if the order is lower than the max order
                    if result_order <= max_order:

                        # Do not run this on main thread
                        new_poly = AugPZonotope \
                            .create_empty_polynomial_zonotope(current_polytope,
                                                              poly_i, n_low,
                                                              e_list[result_order - 1][:, 1:],
                                                              inverse_list[result_order - 1])
                        polytope_queue_all.append(new_poly)

                        # Kill thread after adding the polytope to the queue.
                        # WARNING! Add some flag value that one thread can access the queue at a time
                    else:
                        continue

                # Append the current polytope to the queue
                polytope_queue_all.append(current_polytope)
                if i != max_shifts - 1:
                    result_order = current_polytope.get_exponent() + current_polytope.get_exponent()
                    current_polytope = AugPZonotope.create_empty_polynomial_zonotope(
                        current_polytope,
                        current_polytope, n_low,
                        e_list[result_order - 1][:, 1:],
                        inverse_list[result_order - 1])
            polytope_queue_all.sort(key=lambda x: x.exponent)

            # Assign order of lower dimension
            for poly_i in polytope_queue_all:
                poly_i.set_dim_low(n_low)

            return polytope_queue_all
        else:
            pass

    @staticmethod
    def to_e_keys(e_mat, n_gamma, max_order):
        """
        TODO ???? wat does this do??

        Parameters
        ----------
        e_mat
        n_gamma
        max_order

        Returns
        -------

        """
        base_list = np.array([(max_order + 1) ** i for i in range(n_gamma)]).reshape((-1, 1))
        e_keys = np.sum(np.multiply(e_mat, base_list), axis=0)
        return e_keys

    @staticmethod
    def get_e_inverse(e_new, map_order_to_pos, n_gamma, max_order):
        """
        Get the inverse array used for the compact operation

        Parameters
        ----------
        e_new
        map_order_to_pos
        n_gamma
        max_order

        Returns
        -------

        """
        e_keys = PZonotopeList.to_e_keys(e_new, n_gamma, max_order)
        inverse_array = np.vectorize(map_order_to_pos.get)(e_keys)
        return inverse_array

    @staticmethod
    def _create_e2(n_gamma: int, max_order: int):
        """
        Generate matrices E for the Sparse Polynomial Zonotope
        This function is to generate all unique columns of E. Order is not important

        Parameters
        ----------
        n_gamma : int
            amount of generators in zonotope/polynomial zonotope
        max_order : int
            amount of Lie derivatives that have to be calculated

        Returns
        -------
        Dictionary with lists only the first element contains a numpy array which represents E of the
                     Sparse polynomial zonotope
        """

        monom_list = get_order_list(n_gamma, max_order)
        e_last = np.concatenate((np.zeros((n_gamma, 1), dtype=np.int64), np.array([*monom_list], dtype=np.int64).T),
                                axis=1)
        e_list = [e_last[:, :comb(i + n_gamma, i)] for i in range(1, max_order + 1)]
        inverse_list = [np.ndarray] * max_order
        inverse_list[0] = np.arange(n_gamma + 1)
        e_keys = PZonotopeList.to_e_keys(e_last, n_gamma, max_order)
        generator_position = np.arange(e_last.shape[1])
        map_key_to_position = {e_keys[i]: generator_position[i] for i in range(e_last.shape[1])}

        bool_array = [False] * max_order
        bool_array[0] = True
        max_shifts = int(np.floor(np.log2(max_order)) + 1)
        exponent2_list = [2 ** i for i in range(max_shifts)]

        for exponent2 in exponent2_list:
            bool_array_temp = copy.deepcopy(bool_array)

            # This loop can be ran parallel
            for i in range(0, max_order):
                if not (bool_array_temp[i]):
                    continue
                result_monomial = exponent2 + i + 1
                if result_monomial > max_order:
                    continue
                e_new = PZonotopeList.__kron_e(e_list[exponent2 - 1], e_list[i])

                inverse_list[result_monomial - 1] = PZonotopeList.get_e_inverse(e_new, map_key_to_position, n_gamma,
                                                                                max_order)

                bool_array[result_monomial - 1] = True

        return e_list, inverse_list

    @classmethod
    def from_list(cls, pz_list_in):
        """
        Creates the PolynomialZonotopeList based on a list variable containing Polynomial Zonotopes

        Parameters
        ----------
        pz_list_in

        Returns
        -------

        """
        dim_low = -1
        max_order = -1
        for pz_i in pz_list_in:
            dim_low = getattr(pz_i, 'DimLow', -1)
            if dim_low != -1:
                break

        for pz_i in pz_list_in:
            max_i = getattr(pz_i, 'exponent', -1)
            if max_i > max_order:
                max_order = max_i

        output_obj = cls(pz_list_in, dim_low, max_order)
        return output_obj

    @classmethod
    def generate_list(cls, dim_low: int, max_order: int):
        """
        Constructor of the Monomial transformer

        Parameters
        ----------
        dim_low : int
            Dimension of differential equation
        max_order : int
            Maximum order of monomials

        Returns
        -------

        """

        # List of Polynomial zonotopes

        polynomial_zonotope_list = PZonotopeList._create_polynomial_zonotope_template(dim_low, max_order)
        # set the pointer list for G
        output_obj = cls(polynomial_zonotope_list, dim_low, max_order)

        return output_obj

    def __init__(self, pz_list_in: List[AugPZonotope], dim_low: int, max_order: int):
        self.MaxOrder = max_order
        self.dimension_low = dim_low
        self.polynomial_zonotope_list = pz_list_in
        self.g_list = [Type[AugPZonotope]] * len(pz_list_in)
        self.__set_gmat()

    @staticmethod
    def create_projected_zonotope_list(dim_low: int, max_order: int, n_generators=-1):
        """
        Allocate a list of zonotopes

        Parameters
        ----------
        dim_low : int
            Dimension of the differential equation
        max_order
            Highest order monomial used
        n_generators
            Amount of generators used for the initial set
        Returns
        -------

        """
        if n_generators == -1:
            n_generators = dim_low
        output_list = [Type[AugPZonotope]] * max_order

        for i in range(max_order):
            output_list[i] = AugPZonotope(np.empty((dim_low, 1)), None,
                                          np.empty((dim_low, comb(i + n_generators + 1, i + 1) - 1)),
                                          np.empty((dim_low, comb(i + n_generators + 1, i + 1) - 1)),
                                          is_empty=True, dimension_low=dim_low)

        return output_list

    @staticmethod
    def __kron_e(e_1: np.ndarray, e_2: np.ndarray):
        """
        Structure of the Matrix E after applying the kronecker product of two polynomial zonotopes
            - E1 is the matrix E of the first polynomial zonotope
            - E2 is the matrix  E of the second polynomial zonotope

        Parameters
        ----------
        e_1 : ndarray
            is the matrix E of the first polynomial zonotope
        e_2 : ndarray
            is the matrix  E of the second polynomial zonotope

        Returns
        -------
        output is the matrix of the resulting polynomial zonotope which.
        """

        e_temp_1 = np.repeat(e_1, e_2.shape[1], axis=1)
        e_temp_2 = np.tile(e_2, e_1.shape[1])

        # Add powers alpha^i + alpha^j = alpha^(i+j)
        return e_temp_1 + e_temp_2

    @staticmethod
    def __kronecker_product(spz_1: AugPZonotope, spz_2: AugPZonotope, spz_out: AugPZonotope) -> AugPZonotope:
        """
        In this function the higher state is lifted to order a+b, where a = 2^x and b<=a

        Parameters
        ----------
        spz_1 : AugPZonotope
            polynomial zonotope of order a (2^x)
        spz_2 : AugPZonotope
            polynomial zonotope of order b (b<=a)
        spz_out : AugPZonotope
            polynomial zonotope of order a + b

        Returns
        -------

        """

        # Order of new polytope
        # Get index of dimension
        tail1 = spz_1.get_filtered_tail()
        head2 = spz_2.get_head()
        compress_id = spz_out.compress_array

        # Get generators of each polytope
        g1 = spz_1.get_g()
        g2 = spz_2.get_g()
        gout = spz_out.get_g()

        # Reset output matrix
        gout.fill(0)
        start_index = 0

        # i represents the index of the second input
        for i in np.arange(spz_1.get_dim_low()):
            # Get indices to use of G
            tail1_index = tail1[i]
            head2_index = head2[i]

            # Extra rows memory location
            g1p = g1[tail1_index, :]
            g2p = g2[head2_index, :]

            # Allocate memory for temporary matrix, would be nice to allocate it before the loop and store it in spz out
            g_temp = np.empty((g1p.shape[0] * g2p.shape[0],
                               (g1p.shape[1]) * (g2p.shape[1])))
            # Calculate kronecker product
            g_temp = kronecker_product(g1p, g2p, g_temp)

            n_row_temp = g_temp.shape[0]

            compress_generators(g_temp, compress_id,
                                gout[start_index:(start_index + n_row_temp), :])
            # move last row
            start_index = start_index + g_temp.shape[0]

        gout[:] = gout[spz_out.reorder_array, :]
        return spz_out

    def get_dimension_lifted(self) -> int:
        """
        Get the dimension of the lifted state

        Returns
        -------

        """

        dim_high = 0
        for i in range(len(self.polynomial_zonotope_list)):
            if self.polynomial_zonotope_list[i] is None:
                continue
            dim_high += self.polynomial_zonotope_list[i].g_mat.shape[0]
        return dim_high

    def get_g_list(self) -> List[np.ndarray]:
        """
        Get center + Dependent generators G of all polynomial zonotopes

        Returns
        -------

        """

        return self.g_list

    def lift_n_to_m(self, input_zonotope: Zonotope):
        """
        Lift the state from R^n to R^m

        Parameters
        ----------
        input_zonotope : Zonotope
            Xk

        Returns
        -------
        List of PolynomialZonotope which is the lifted state
        """

        polynomial_zonotope_0 = input_zonotope.as_augmented_polynomial_zonotope()
        polynomial_zonotope_0.empty_gi()
        polynomial_zonotope_0.set_dim_low(self.dimension_low)
        polynomial_zonotope_0.set_exponent(1)
        PZonotopeList.__update_lifted_state(polynomial_zonotope_0, self.polynomial_zonotope_list)

        # Quick fix adjust pointer location
        # self.__update_gmat(self.polynomial_zonotope_list)

        return self.polynomial_zonotope_list

    def __len__(self) -> int:
        """
        Get the amount of Polynomial Zonotopes in list

        Returns
        -------
        Amount of Polynomial zonotopes
        """
        return len(self.polynomial_zonotope_list)

    def __set_gmat(self) -> None:
        """
        Saves all pointers of matrices G to one list

        Returns
        -------

        """
        n_el = len(self.polynomial_zonotope_list)
        self.g_list = [None] * n_el
        for i in range(n_el):
            self.g_list[i] = self.polynomial_zonotope_list[i].g_mat

    def __update_gmat(self, polynomial_zonotope_list: List[AugPZonotope]) -> None:
        for i in range(len(polynomial_zonotope_list)):
            self.g_list[i] = polynomial_zonotope_list[i].g_mat

    @staticmethod
    def __update_lifted_state(spz_base: PZonotope, template_monomials):
        """
        In this function the initial set is lifted to a set of polynomial zonotopes or monomials of order i.
        Since the amount of generators are not equal to each other the sets are not concatenated.

        Parameters
        ----------
        spz_base : PZonotope
            Initial state in the original coordinate system
        template_monomials
            Structure + allocated memory for the lifted space

        Returns
        -------

        """

        # TODO Reduce observer by not calculating unused monomials

        spz_base.empty_gi()
        max_order = len(template_monomials)

        # Amount of leading polytopes. This is equal to the highest 2^x < max order
        max_shifts = int(np.floor(np.log2(max_order)) + 1)
        exponent2_list = [2 ** i for i in range(max_shifts)]
        bool_array = np.array([False] * max_order)
        bool_array[0] = True

        # Set first value of list
        np.copyto(template_monomials[0].g_mat, spz_base.g_mat)
        # This loop cannot be ran parallel
        for exponent2 in exponent2_list:

            bool_array_temp = copy.deepcopy(bool_array)
            # bool_array = [False] * N

            # This loop can be ran parallel
            for i in nb.prange(max_order):
                if not (bool_array_temp[i]):
                    continue
                result_monomial = exponent2 + i + 1
                if result_monomial > max_order:
                    continue

                PZonotopeList.__kronecker_product(template_monomials[exponent2 - 1],
                                                  template_monomials[i],
                                                  template_monomials[result_monomial - 1])
                bool_array[result_monomial - 1] = True
                if np.all(bool_array):
                    break
            if np.all(bool_array):
                break
        return

    def to_zonotope(self, zonotope_out=None) -> Union[List[Zonotope], Zonotope]:
        """
        Converts the Polynomial Zonotope List to a list of zonotopes

        Parameters
        ----------
        zonotope_out : Zonotope
            allocated memory for output
        Returns
        -------

        """

        n_list = len(self.polynomial_zonotope_list)
        output_list = []

        for i in range(n_list):

            # If there is not
            if zonotope_out is None:
                output_list.append(self.polynomial_zonotope_list[i].to_zonotope())
            else:
                self.polynomial_zonotope_list[i].to_zonotope(zonotope_out[i])

        # Returning the value
        if zonotope_out is None:
            return output_list
        else:
            return zonotope_out

    def get_over_approx_inf(self) -> float:
        """
        Get the over-approximated infinity norm of a polynomial zonotope list
        Returns
        -------

        """
        inf_norm = 0
        for i in range(len(self.polynomial_zonotope_list)):
            inf_new = self.polynomial_zonotope_list[i].get_over_approx_inf()
            if inf_norm < inf_new:
                inf_norm = inf_new

        return inf_norm

    def transform_to(self, transform_matrix: np.ndarray, pz_out: AugPZonotope) -> AugPZonotope:
        """
        Transforms each polynomial zonotope in the list to the next time step by multiplying using C exp(Kt)
        After transforming these polynomial zonotopes. All polynomial zonotopes are summed up
        In this function the polynomial zonotope list is transformed

        Parameters
        ----------
        transform_matrix : ndarray
            List of transform matrices with dimension R^{n x k} with >= n
        pz_out : AugPZonotope
            Allocated memory for output
        Returns
        -------
        Transformed pz_list_obj_2, which is the projected state at time step k+1
        """

        # get pointer for generator matrix
        pz_list_1 = self.g_list
        pz_out_g = pz_out.g_mat
        pz_out_g.fill(0)
        n_lists = len(self.g_list)

        # Iterate over the the output polynomial zonotope list
        # TODO parallelize i-loop. Create a list of the amount of rows for each polynomial zonotope

        start_ind_j = 0
        # Iterate over all input polynomial zonotopes
        for j in range(n_lists):
            if pz_list_1[j] is None:
                continue

            pz_out_g[:, :pz_list_1[j].shape[1]] += \
                transform_matrix[:pz_out_g.shape[0], start_ind_j:(start_ind_j + pz_list_1[j].shape[0])] \
                    .dot(pz_list_1[j]
                         )
            start_ind_j += pz_list_1[j].shape[0]
        return pz_out
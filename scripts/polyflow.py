"""
    This file contains function / classes to get the Polyflow operator / error bound
"""
from typing import Type, Tuple
from typing import List

import json
import cvxpy as cp
import numpy as np
import numba as nb
from scipy.linalg import expm
from scipy import optimize
from sympy.printing.aesaracode import aesara_function
from sympy import Matrix, Poly, symbols
from sympy.core.symbol import Symbol
from sympy.polys.polymatrix import PolyMatrix
from scripts.misc_functions import get_carleman_to_poly
from scripts.dreal_error_bound import DrealErrorBound as De
# from scripts.dreal_error_bound import Z3ErrorBound as Ze


class Domain:
    """
    This class describes the domain of Polyflow.
    It auto generates the mesh grid for a given boundary set and step size
    """

    center: np.ndarray
    axis_length: np.ndarray
    dim_low: int

    def __init__(self, axes_desc: np.ndarray):
        """ The constructor of Domain class """

        self.grid_list = self.__generate_grid(axes_desc)
        self.dim_low = len(self.grid_list)
        self.bounds = axes_desc[:, :2]
        self.center = np.sum(self.bounds, axis=1).reshape((-1, 1)) / 2
        self.axis_length = np.abs(axes_desc[:, 0].reshape((-1, 1)) - self.center)
        self.axes_desc = axes_desc

    def get_box(self, doi=None) -> List[List[float]]:
        """ Get the projected hyper rectangle in the specified plane.

        Parameters
        ----------
        doi : List[int]
            indices of the plane of interest
        """

        if doi is None:
            doi = [0, 1]
        return [[self.bounds[doi[0], 0], self.bounds[doi[0], 1], self.bounds[doi[0], 1], self.bounds[doi[0], 0],
                 self.bounds[doi[0], 0]],
                [self.bounds[doi[1], 0], self.bounds[doi[1], 0], self.bounds[doi[1], 1], self.bounds[doi[1], 1],
                 self.bounds[doi[1], 0]]]

    def get_bounds(self) -> np.ndarray:
        """ Returns the lower and upper bound of each element of the domain """
        return self.bounds

    def get_grid(self) -> tuple:
        """ Returns the mesh grid of the domain for each dimension"""
        return self.grid_list

    def get_n_points(self) -> int:
        """ Returns the amount of points of the grid """
        return len(self.grid_list[0])

    def get_dim_low(self) -> int:
        """ Returns the dimension of the system """
        return self.dim_low

    def get_center(self) -> np.ndarray:
        """ Returns the center of the domain """
        return self.center

    def to_dict(self) -> dict:
        """ Converts the domain object to a dictionary """
        return {'domain': self.axes_desc.tolist()}

    def to_json(self) -> str:
        """ Converts the Domain object to a string in json format. """
        return json.dumps(self.to_dict())

    @staticmethod
    def __generate_grid(domain_description_in: np.ndarray) -> tuple:
        """
        This function generates all points of the grid
        The output is a tuple. Each elements contains all values of the respective dimension

        Parameters
        ----------
        domain_description_in:
            description of the grid where each row is defined as [left bound, right bound, stepsize]

        Returns
        -------
        List of tuples which contain the coordinates of the mesh grid
        """

        n = domain_description_in.shape[0]  # Get dimension of system
        grid_tuple = (None,)  # Initiate Tuple

        for i in range(0, n):
            grid_tuple += (np.arange(domain_description_in[i, 0],
                                     domain_description_in[i, 1] + domain_description_in[i, 2],
                                     domain_description_in[i, 2]),)

        mesh = np.array(np.meshgrid(*grid_tuple[1:]))  # All grid points excluding the None variable

        return tuple(mesh.T.reshape(-1, n).T)  # Reshape matrix to get an n x m matrix


class PolyFlow:
    """
    PolyFlow class is used to define the linear matrix
    """

    operator = None
    carl_to_poly_reduced = None

    @staticmethod
    def __evaluate_lie_derivatives(lie_list_in: List[aesara_function], domain_obj: Domain) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates all Lie derivatives over all grid points

        Parameters
        ----------
        lie_list_in
            variable containing theano functions of the lie derivatives
        domain_obj
            variable containing the grid values
        """

        m = len(lie_list_in)  # amount of Lie functions
        n = len(domain_obj.get_grid())  # dimension of system,
        grid_n = len(domain_obj.get_grid()[0])  # Amount of grid points in domain

        # Allocate space for Matrix A
        known_lie = np.zeros((n * grid_n, m - 1))

        # Evaluate all {0, N-1} Lie derivatives at the grid points
        for i in range(0, m - 1):
            known_lie[:, i] = lie_list_in[i](*domain_obj.get_grid()).reshape((n, -1)).T.reshape((-1, 1)).ravel()

        # Get Nth Lie derivative which is to be approximated by polyflow
        to_be_estimated_lie = lie_list_in[-1](*domain_obj.get_grid()).reshape((n, -1)).T.reshape((-1, 1))

        return known_lie, to_be_estimated_lie

    def __get_all_lie_derivatives(self, diff_function_in: PolyMatrix, sym_list: Tuple[symbols],
                                  max_derivative: int) -> Tuple[list, List[PolyMatrix]]:
        """
        Calculates all Lie derivatives from 0 to max_derivative
        The next Lie derivative are obtained by using __get_next_lie_derivative.
        After the Lie derivative it is converted to an aesara function for fast evaluation speed.

        Parameters
        ----------
        diff_function_in
            Matrix containing polynomial symbolic functions
        sym_list
            Matrix containing all symbolics of the differential equation.
            Order has to be the same as in the differential function
        max_derivative
            The amount of Lie derivatives used for the Polyflow and is equal

        Returns
        -------
        All Lie derivatives up to order max_derivative
        """

        # Initiate list
        lie_derivative_aesara_list = [aesara_function] * (max_derivative + 1)
        lie_derivative_sympy_list = [Type[PolyMatrix]] * (max_derivative + 1)

        # Set first 0th Lie derivative as current function
        current_lie = PolyMatrix(sym_list)

        # Create dictionary for theano function all symbolics have dimension
        dims_arg = dict((key_i, 1) for key_i in sym_list)
        dtype_arg = dict((key_i, 'float64') for key_i in sym_list)

        # Get Lie derivative function for 0th order
        lie_derivative_aesara_list[0] = aesara_function(sym_list, [current_lie], dims=dims_arg, dtypes=dtype_arg)
        lie_derivative_sympy_list[0] = PolyMatrix([Poly(current_lie[i], sym_list) for i in range(0, len(current_lie))])

        # Set first function as current function
        current_lie = lie_derivative_sympy_list[0]

        # Get higher order lie derivatives and create theano function of it
        for i in range(1, max_derivative + 1):
            current_lie = self.__get_next_lie_derivative(current_lie, diff_function_in, sym_list)
            current_func_non_poly_obj = Matrix([current_lie[j].as_expr() for j in range(0, len(sym_list))])
            lie_derivative_aesara_list[i] = aesara_function(sym_list, [current_func_non_poly_obj], dims=dims_arg,
                                                            dtypes=dtype_arg)
            lie_derivative_sympy_list[i] = current_lie

        return lie_derivative_aesara_list, lie_derivative_sympy_list

    @staticmethod
    def __get_next_lie_derivative(current_function: PolyMatrix, f: PolyMatrix, diff_symbols: Tuple[symbols]) \
            -> PolyMatrix:
        """
        Calculates the next Lie Derivative of the input by taking the Jacobian of the function and multiplying it with
        the differential equation.

        Parameters
        ----------
        current_function
            k-1 th Lie derivative
        f
            Differential equation of the nonlinear system
        diff_symbols
            Symbolics of the differential equation
        Returns
        -------
        k th Lie derivative
        """

        m1 = current_function.jacobian(diff_symbols)
        return m1 * f

    def to_dict(self) -> dict:
        """ Wraps the Polyflow object in a dictionary which is compatible with json format """
        output_dict = {}
        key_list = ['input_differential_eq', 'symbol_tuple', 'domain_obj', 'max_lie_order', 'time_step',
                    'scale_factor', 'extra_eig', 'bloat_scale', "polyflow_error"]
        for key_i in key_list:
            output_dict.update(to_json_el(self, key_i))
        output_dict.update(self.get_overrides())
        return output_dict

    # Type hinting for PyCharm
    lie_sympy_list: List[PolyMatrix]
    symbol_tuple: Tuple[symbols]
    domain_obj: Domain
    polyflow_error: np.ndarray
    continuous_matrix_list: List[np.ndarray]
    scale_factor: float
    from_dict_bool: bool
    extra_eig: float
    solver: str
    time_step: float
    smt_solver: str
    polyflow_smt_tol: List[float]
    operator_list = [None]

    @staticmethod
    def _create_scale_list(scale_factor, max_lie_order):
        """ Create scale list for coordinate transformation """
        return np.array([scale_factor ** -(max_lie_order - i - 1) for i in range(max_lie_order)])

    def _init_cvx_problems(self, input_differential_eq, symbol_tuple, max_lie_order,
                           domain_obj, dim_low, extra_eig, scale_factor):
        """ Create CVX object in order to optimize the Lambda values of the Polyflow """


        lie_list, lie_sympy_list = self.__get_all_lie_derivatives(input_differential_eq, symbol_tuple,
                                                                  max_lie_order)
        # Evaluate Lie derivatives for optimization problem
        known_lie, to_be_estimated_lie = self.__evaluate_lie_derivatives(lie_list, domain_obj)

        # update constraints for optimization problem
        model_list, var_list = self.__get_cvx_obj(dim_low, known_lie, to_be_estimated_lie,
                                                  extra_eig, scale_factor)
        return lie_list, lie_sympy_list, known_lie, \
               to_be_estimated_lie, model_list, var_list

    def __init__(self, input_differential_eq: PolyMatrix, symbol_tuple: Tuple[symbols],
                 domain_description_in: np.ndarray, max_lie_order: int, time_step: float,
                 **kwargs):
        """
        Constructor of class object PolyFlow

        Parameters
        ----------
        input_differential_eq
            Differential equation of the nonlinear system
        symbol_tuple
            All symbolics of the differential equation
        domain_description_in
            Description of the domain
        max_lie_order
            The order of the Lie derivative that is to be estimated
        time_step
            Time step of the reachability algorithm
        kwargs
            from_dict_bool
            lie_sympy_list
            lambda_list
            polyflow_error_factors
            exponent_factors
            bloat_scale
            scale_factor
                Factor used for the coordinate transformation
            extra_eig
                Relaxation of the eigen value constraint. This variable decides how much the spectral radius may be above
                the scaling factor spectral_allowed = scale_factor*(1 + extra_eig)

        """

        prop_defaults = {'from_dict_bool': False,
                         'solver': 'SCS',
                         'smt_solver': 'dreal',
                         'map_matrix': None,
                         'lambda_variable_matrices': None,
                         'scale_factor': 1.0,
                         'extra_eig': 0.2,
                         'lambda_list': [Type[np.ndarray], Type[np.ndarray]],
                         'projection_matrix': np.empty((5, 10)),
                         'flowpipe_smt_tol': None,
                         'polyflow_smt_tol': None,
                         'model_list': [],
                         'operator_list': Type[list]
                         }

        # Set variables with default argument
        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
            if prop in kwargs.keys():
                kwargs.pop(prop)

        # Set variables
        for key_i, value in kwargs.items():
            setattr(self, key_i, value)

        # Set necessary defined variables
        self.time_step = time_step
        self.max_lie_order = max_lie_order
        self.input_differential_eq = input_differential_eq
        self.symbol_tuple = symbol_tuple

        self.scale_list = self._create_scale_list(self.scale_factor, max_lie_order)
        dim_low = len(symbol_tuple)

        # initial value of line search for the Polyflow error
        self.min_error = np.zeros(dim_low)

        # Domain variable (necessary coupling)
        self.domain_obj = Domain(domain_description_in)

        if not self.from_dict_bool:

            # Get all Lie derivatives aesara function/symbolic
            self.lie_list, self.lie_sympy_list = self.__get_all_lie_derivatives(input_differential_eq, symbol_tuple,
                                                                                max_lie_order)
            # Evaluate Lie derivatives for optimization problem
            self.known_lie, self.to_be_estimated_lie = self.__evaluate_lie_derivatives(self.lie_list, self.domain_obj)

            # update constraints for optimization problem
            self.model_list, self.var_list = self.__get_cvx_obj(dim_low,
                                                                self.known_lie,
                                                                self.to_be_estimated_lie,
                                                                self.extra_eig,
                                                                self.scale_factor)
            self.lie_list, self.lie_sympy_list, self.known_lie, \
            self.to_be_estimated_lie, self.model_list, self.var_list = \
                self._init_cvx_problems(input_differential_eq, symbol_tuple, max_lie_order,
                                        self.domain_obj, dim_low, self.extra_eig, self.scale_factor)

        # Get order of highest monomial of the N - 1th Lie derivative
        self.max_monomial_order = get_max_order_observer(self.lie_sympy_list[-2])

        # Get matrix which maps the monomials to polynomials
        self.carl_to_poly, _ = get_carleman_to_poly(self.lie_sympy_list[:-1], symbol_tuple)

        self.continuous_matrix, self.discrete_matrix, self.operator, self.continuous_matrix_list = \
            self._allocate_matrices_memory(dim_low, max_lie_order, self.carl_to_poly.shape[1])

        # Allocate memory for each polyflow operator 
        for i in range(dim_low):
            sub_matrix = np.zeros((max_lie_order, max_lie_order))
            sub_matrix[:-1, 1:] = np.eye(max_lie_order - 1) * self.scale_factor
            self.continuous_matrix_list.append(sub_matrix)

        if not self.from_dict_bool:
            # Solve the linear problem and obtain lambda
            self.lambda_list, self.continuous_matrix_list, \
            self.lambda_variable_matrices, self.continuous_matrix, \
            self.discrete_matrix, self.operator = self.solve_lambda(dim_low, self.solver,
                                                                    self.model_list,
                                                                    max_lie_order,
                                                                    time_step, self.operator,
                                                                    self.scale_factor,
                                                                    self.continuous_matrix,
                                                                    self.continuous_matrix_list)
            # Determine new scale factors
            self.get_scale_factors_bloating()

            # Determine the error bound object for for the polyflow
            array_2norm = self.get_2norms(self.continuous_matrix_list)
            self.spectral_radii = array_2norm
            if self.smt_solver == 'dreal':
                # TODO add lambda keyword for set_lambda
                self.error_obj = De(self.lie_sympy_list, symbol_tuple, domain_description_in[:, :2].tolist(),
                                    self.min_error, self.scale_factor, time_step, array_2norm, self.polyflow_smt_tol,
                                    )
            elif self.smt_solver == 'z3':
                raise NotImplementedError
                # # TODO CHANGE
                # self.error_obj = Ze(self.lie_sympy_list, symbol_tuple, domain_description_in[:, :2].tolist(),
                #                     self.min_error)

            self.error_obj.set_lambda(self.lambda_list)  # Set lambda parameters
            self.bloat_scale = self.get_scale_factors_bloating()  # Determine new scale factors
            self.polyflow_error, self.polyflow_error_factors = self.error_obj.calculate_error_bounds()

            # TODO RENAME this exponent factor variable (REMOVE?)
            exp_kt = np.diag([np.exp(array_2norm[i] * time_step) for i in range(len(array_2norm))])
            self.exponent_factor = time_step * self.scale_factor ** (-max_lie_order + 1) * exp_kt
        else:
            raise NotImplementedError
        print(self.polyflow_error)

        # Get matrix which maps the monomials to polynomials with coordinate transformation
        self.carl_to_poly_reduced = np.diag(np.array([self.bloat_scale ** (-i) for i in range(max_lie_order)]).
                                            repeat(dim_low)) * self.carl_to_poly

    @staticmethod
    def _allocate_matrices_memory(dim_low, max_lie_order, n_monomials):
        """
        Allocates memory for the Polyflow operators

        Parameters
        ----------
        dim_low
            Dimension of differential equation
        max_lie_order
            Amount of observers per sub system
        n_monomials
            maximum amount of monomials that a N-1 Lie derivative can have
        """

        operator_size = dim_low * max_lie_order

        continuous_matrix = np.zeros((operator_size,
                                      operator_size))  # Continuous Polyflow operator memory
        continuous_matrix[:-dim_low,
        dim_low:] = np.eye(operator_size - dim_low)  # Set integrator

        discrete_matrix = np.zeros((operator_size,
                                    operator_size))  # Discrete Polyflow operator memory (R^m -> R^m)
        operator = np.empty((dim_low, n_monomials))  # Discrete Polyflow operator memory (R^m -> R^n)
        continuous_matrix_list = []  # Holds Polyflow sub-systems

        return continuous_matrix, discrete_matrix, operator, continuous_matrix_list

    def get_scale_factors_bloating(self) -> float:
        """ Get the minimum scale factor for which all sub systems of the Polyflow have to lowest spectral radius """

        self.operator_list = [np.zeros(self.continuous_matrix_list[0].shape) for _ in range(len(self.lambda_list))]
        scale_candidates = np.zeros(len(self.lambda_list))
        for index in range(len(self.lambda_list)):
            # TODO might have to increase lower bound depending on domain to reduce the infinity norm properly
            scale_0 = np.array([self.scale_factor])
            result = optimize.minimize(self.get_scale_cost, scale_0, args=(index,), bounds=((1, None),))
            scale_candidates[index] = result['fun']

        return np.max(scale_candidates)

    def get_scale_cost(self, scale_factor: np.ndarray, index: int) -> float:
        """
        Cost function for the bloating problem
        Parameters
        ----------
        scale_factor : ndarray
            Input value which is the scaling factor on the
        index
            Entry of the problem
        Returns
        -------
        Cost value which is equal to the spectral radius
        """
        scale_factor = float(scale_factor)

        # Create Polyflow operator
        self.operator_list[index][:-1, 1:] = np.eye(self.max_lie_order - 1) * scale_factor
        self.operator_list[index][-1, :] = np.array([self.lambda_list[index][i] * (scale_factor **
                                                                                   -(self.max_lie_order - 1 - i)) for
                                                     i in range(self.max_lie_order)])
        poly_2norm = self.get_2norm(self.operator_list[index])
        return poly_2norm

    @staticmethod
    def get_2norm(matrix_in: np.ndarray) -> float:
        """
        Get the spectral radius of the a matrix by taking the maximum absolute eigenvalue found with SVD

        Parameters
        ----------
        matrix_in : ndarray
            continuous matrix of the polyflow
        Returns
        -------
        2 norm of Polyflow operator
        """

        return np.max(np.abs(np.linalg.svd(matrix_in, compute_uv=False)))

    def get_2norms(self, matrix_list: List[np.ndarray]) -> List[float]:
        """
        Get all spectral radii of all continuous matrices of the polyflow
        Parameters
        ----------
        matrix_list
            List of polyflow matrices
        Returns
        -------
        spectral radius of all matrices
        """
        return [self.get_2norm(matrix_list[i]) for i in range(len(matrix_list))]

    @staticmethod
    def _get_polyflow_abs_constraints(known_lie, to_be_estimated_lie, variables):
        n_rows = known_lie.shape[0]
        abs_up = np.concatenate((known_lie, -np.ones((n_rows, 1))), axis=1) \
                 @ variables <= to_be_estimated_lie.flatten()
        abs_low = np.concatenate((-known_lie, -np.ones((n_rows, 1))), axis=1) \
                  @ variables <= -to_be_estimated_lie.flatten()
        return abs_up, abs_low

    @staticmethod
    def _get_polyflow_infinity_norm_constraints(max_order, extra_eig, scale_factor, variables):
        inf_norm_up = np.concatenate((np.eye(max_order - 1), np.zeros((max_order - 1, 1))), axis=1) \
                      @ variables <= np.array([(1 + extra_eig) * scale_factor
                                               if ii == 0 else extra_eig * scale_factor for ii in
                                               range(max_order - 1)])
        inf_norm_low = -np.concatenate((np.eye(max_order - 1), np.zeros((max_order - 1, 1))), axis=1) \
                       @ variables <= np.array([(1 + extra_eig) * scale_factor
                                                if ii == 0 else extra_eig * scale_factor for ii in
                                                range(max_order - 1)])
        return inf_norm_up, inf_norm_low

    def __get_cvx_obj(self, lower_dimension, known_lie, to_be_estimated_lie, extra_eig, scale_factor):
        """ Updates the linear inequality constraints of the linear problem. (slack variables, infinity norm) """

        max_order = known_lie.shape[1] + 1  # Estimated Lie order
        n_rows = known_lie.shape[0]  # grid points * dimension
        model_list = [Type[cp.Problem]] * lower_dimension
        polyflow_variables = [Type[cp.Variable]] * lower_dimension
        cost_matrix = np.array([0, ] * (max_order - 1) + [1, ])

        for k in range(lower_dimension):
            set_j = np.arange(k, n_rows, lower_dimension)

            polyflow_variables[k] = cp.Variable(max_order)
            sub_problem_variables = polyflow_variables[k]

            abs_up, abs_low = self._get_polyflow_abs_constraints(known_lie[set_j, :],
                                                                 to_be_estimated_lie[set_j, :],
                                                                 sub_problem_variables)
            inf_norm_up, inf_norm_low = self._get_polyflow_infinity_norm_constraints(max_order,
                                                                                     extra_eig,
                                                                                     scale_factor,
                                                                                     sub_problem_variables)
            model_list[k] = cp.Problem(cp.Minimize(cost_matrix @ sub_problem_variables),
                                       [abs_up, abs_low,
                                        inf_norm_up, inf_norm_low])

        return model_list, polyflow_variables

    @staticmethod
    @nb.njit(fastmath=True, parallel=True)
    def reshape_lie_matrix(known_lie, n, n_grid, max_order) -> np.ndarray:
        """
        Reshapes the the matrix of known values to a shape that can be used for the optimization problem

        Parameters
        ----------
        known_lie : ndarray
            List of grid values of the Nth Lie derivative
        n : int
            dimension of the problem
        n_grid :int
            amount of grid points
        max_order : int
            Amount of Lie derivatives
        """
        output_matrix = np.zeros((n * max_order, n_grid), np.float64)

        for i in nb.prange(n_grid):
            for j in nb.prange(max_order):
                for k in nb.prange(n):
                    output_matrix[n * j + k, i] = known_lie[n * i + k, j]
        return output_matrix

    def solve_lambda(self, dim_low, solver, model_list, max_lie_order, time_step, operator,
                     scale_factor, continuous_matrix, continuous_matrix_list):
        """
            In this function all Lambda parameters are optimized over a finite grid.
            After solving the lambda parameters. these lambda parameters are stored in self.lambda_list

        Parameters
        ----------
        dim_low
            Dimension of system
        solver
            Name of CVX solver
        model_list
            List containing the CVX models
        max_lie_order
            Maximum lie order
        time_step
            Time step of reachability
        operator
            Operator of the Polyflow
        scale_factor
            Factor used for coordinate transformation
        continuous_matrix
            Memory for continuous time matrix of Polyflow
        continuous_matrix_list
            List of continuous time matrices of each subsystem
        """

        n_models = dim_low
        lambda_list = [np.empty(max_lie_order)] * n_models
        lambda_variable_matrices = []
        discrete_matrix = None
        if model_list is not []:

            for i in range(n_models):
                print('Getting lambda')

                model_list[i].solve(solver=getattr(cp, solver))
                variables = model_list[i].variables()[0]
                all_values = variables.value
                lambda_list[i] = all_values[:-1]
                self.min_error[i] = all_values[-1]

            lambda_variable_matrices, continuous_matrix, continuous_matrix_list, \
            discrete_matrix, operator = self.set_lambda_values(lambda_list, max_lie_order, dim_low, scale_factor,
                                                               continuous_matrix, continuous_matrix_list,
                                                               operator, time_step)

        else:
            print('no list found')
            raise ValueError

        return lambda_list, continuous_matrix_list, \
               lambda_variable_matrices, continuous_matrix, discrete_matrix, operator

    def set_lambda_values(self, lambda_list: list, max_order, n, scale_factor, continuous_matrix,
                          continuous_matrix_list, operator, time_step):
        """
            Updates the values of the polyflow matrix
        Parameters
        ----------
        max_order
        n
        scale_factor
        continuous_matrix
        continuous_matrix_list
        operator
        time_step
        lambda_list : List
            List of lambda parameters which give the linear combination of the first N-1 Lie derivatives
        """

        estimated_matrix = np.zeros((n, n))
        output_matrix = np.zeros((n, n * max_order))

        for i in range(max_order):
            for j in range(n):
                estimated_matrix[j, j] = lambda_list[j][i]
                continuous_matrix_list[j][-1, i] = lambda_list[j][i] * \
                                                   (scale_factor ** -(max_order - 1 - i))

            output_matrix[:, i * n:((i + 1) * n)] = estimated_matrix
        lambda_variable_matrices = output_matrix
        continuous_matrix[-n:, :] = output_matrix
        discrete_matrix = expm(continuous_matrix * time_step)
        np.dot(discrete_matrix[:n, :], self.carl_to_poly.toarray(), out=operator)

        return lambda_variable_matrices, continuous_matrix, continuous_matrix_list, discrete_matrix, operator

    def get_overrides(self) -> dict:
        """ Get all variables required to construct the Polyflow """

        keynames = ['from_dict_bool', 'lie_sympy_list', 'lambda_list',
                    'polyflow_error_factors', 'exponent_factor']

        overrides_dict = {}
        for key_i in keynames:
            overrides_dict.update(to_json_el(self, key_i))

        return {'overrides': overrides_dict}

    @staticmethod
    def parse_lambda(lambda_list: List[List[float]]) -> List[np.ndarray]:
        """ Parses a 2D list of lambda to List[ndarray] """
        return [np.array(lambda_i) for lambda_i in lambda_list]


def get_max_order_observer(lie_in: PolyMatrix) -> int:
    """
    Get maximum order of the monomials of the highest Lie derivative
    Parameters
    ----------
    lie_in
        symbolic expression of the Lie derivative
    Returns
    -------
    maximum order of the monomials
    """
    n = len(lie_in)
    return max([max(map(sum, lie_in[x].monoms())) for x in range(0, n)])


def to_json_el(self_variable, key_name):
    """
    In this
    Parameters
    ----------
    self_variable
    key_name

    Returns
    -------

    """
    try:
        attr = getattr(self_variable, key_name)
    except Exception as e:
        print('key_name: %s has failed' % key_name)
        print(e)
        return
    if type(attr) == Domain:
        return attr.to_dict()
    if type(attr) == np.ndarray:
        attr = attr.tolist()
    elif type(attr) == PolyMatrix:
        attr = str(attr)
    elif type(attr) == list:
        if type(attr[0]) == np.ndarray:
            attr = [attr_i.tolist() for attr_i in attr]
        elif type(attr[0]) == PolyMatrix:
            attr = [str(attr_i) for attr_i in attr]
    elif type(attr) == tuple:
        if type(attr[0]) == Symbol:
            attr = str(attr)

    return {key_name: attr}
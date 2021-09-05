"""
    This file contains classes which are related to bounding an error over
    an interval using an SMT-solver.
"""
import math

import dreal as dr
import numpy as np
from scripts.misc_functions import get_carleman_to_poly, get_smt_equation
from scipy.sparse import csr_matrix
import json
import kill_timeout


class SMTBase(object):
    """
    A class, which basic operation for classes which use SMT techniques to find bounds.

    Attributes
    ----------
    error : List[float]
        List of
    n_dim : int
        Dimension of the system
    variable_list : List[...]
        List of SMT variables used for the SMT equations

    Methods
    -------
    smt_abs
        Function which defines the absolute function for an SMT solver
    get_next_state
        Gives the next time step based on the SMT output
    get_variable_list
        Returns the SMT variables of the problem
    calculate_errors
        Calculates the over-approximation of the maximum absolute difference between the exact and approximated Lie
        derivative (all entries)
    calculate_error_bounds
        Calculates the error bound of the Polyflow (all entries)
    calculate_error
        Calculate the over-approximation of the maximum absolute difference between the exact and approximated Lie
        derivative (one element)
    smt_line_search
        Calculates an over-approximation of the maximum absolute difference between the Nth with a line search
        (one element)

    """

    def __init__(self, n, *args, **kwargs):
        """
        Parameters
        ----------
        n : int
            Dimension of the system
        args
        kwargs
        """
        self.error = np.empty(1)
        self.time_error = None
        self.n_dim = n
        self.variable_list = self.get_variable_list(n)
        super(SMTBase, self).__init__(*args, **kwargs)

    @staticmethod
    def smt_abs(x):
        """
        Function which defines the absolute function for an SMT solver

        Parameters
        ----------
        x : ...
            Argument within the absolute function
        """
        raise NotImplementedError

    @staticmethod
    def get_next_state(res, xk, step, unsat_found):
        """
        Output handler of the line search algorithm per iteration
            If res is 1, there is a satisfied answer found
            If res is 0, the SMT solver could not found a solution
            If res is -1, the SMT solver concluded the problem is unsatisfied
        The next step is determined as follows
            If res is 1 the step size is doubled
            If res is 0 the step size is multiplied with 1.5
            If res is -1 then the step halved
        Parameters
        ----------
        res : Boolean
            Exit flag of the SMT solver
        xk : float
            Current upperbound of the Nth Lie derivative
        step : float
            Step size of the line search
        unsat_found :
            Upperbound for which the SMT solver says "unsatisfied"
        Returns
        -------
        xk : float
            Current upperbound of the Nth Lie derivative
        step : float
            Step size of the line search
        unsat_found :
            Upperbound for which the SMT solver says "unsatisfied"
        """

        if res == 1:
            xk += step
            if unsat_found == -1:
                step *= 2
        elif res == -1:
            unsat_found *= 0
            unsat_found += xk + step
            step /= 2
        elif res == 0:
            if unsat_found == -1:
                step *= 1.5
            else:
                step *= 1.5
        else:
            print('error')

        return xk, step, unsat_found

    @staticmethod
    def get_variable_list(n):
        """
        Returns the SMT variables of the problem

        Parameters
        ----------
        n : int
            Dimension of system
        """
        raise NotImplementedError

    def calculate_errors(self):
        """
        Calculates the errorbound for a fixed domain.
        The error is calculated separately for each entry.

        Returns
        -------

        """

        error_out = np.zeros(self.n_dim)

        for i in range(self.n_dim):
            error_out[i] = self.calculate_error(i)

        self.error = error_out
        return error_out

    def calculate_error_bounds(self):
        """
        Calculates the error bound of the Polyflow (all entries)

        First the Maximum absolute error between the exact and approximated solution.
        After that the Maximum error is multiplied with "t * exp(rho *t)" to obtain the polyflow error
        Returns
        -------

        """
        error = max(self.calculate_errors())
        return np.diag(error * self.time_error), error

    def calculate_error(self, index, timeout=100):
        """
        Over-approximates the maximum absolute error between the exact and approximated Nth Lie derivative
        Parameters
        ----------
        index : int
            ith entry of the dimension
        timeout
            maximum time spend for the SMT solver in milliseconds
        """
        raise NotImplementedError

    def smt_line_search(self, interval_constraints, error_func, left_bound, smt_tolerance, timeout=3.0,
                        max_iterations=10,
                        method='variable-stepsize'):
        """
        Searches for the maximum absolute error of the SMT problem
        Parameters
        ----------
        smt_tolerance
        timeout
        interval_constraints : ndarray
            Description of the domain
        error_func : SMT equation
            Function of the difference between the Nth Lie derivative and its approximation
        left_bound : float
            Left side of the searching area
        max_iterations : int
            Maximum amount of iterations if there is an "unsatisfied" answer found
        method : str
            Method of how the step size is chosen
        """
        raise NotImplementedError


class DrealBase(SMTBase):
    """
    Class which contains basic operation for classes which use DReal as SMT solver.

    Attributes
    ----------
        smt_monoms : ....
            List of all variables which may occur in the smt equation

    Methods
    -------
    get_smt_monomials
        Returns a list of all monomials which may occur in the SMT equation
    """

    def calculate_error(self, index, timeout=100):
        """
        Calculates the error bound of the Polyflow (all entries)

        First the Maximum absolute error between the exact and approximated solution.
        After that the Maximum error is multiplied with "t * exp(rho *t)" to obtain the polyflow error
        Returns

        Parameters
        ----------
        index : int
            entry of the system
        timeout : float
            Amount seconds allowed for the SMT solver
        Returns
        -------

        """
        pass

    order_list: np.ndarray

    def __init__(self, *args, **kwargs):
        super(DrealBase, self).__init__(*args, **kwargs)
        self.smt_monoms = self.get_smt_monomials(self.variable_list, self.order_list)

    @staticmethod
    def get_smt_monomials(variable_list, order_tuple_in):
        """
        Returns a list of all monomials which may occur in the SMT equation.

        -It converts the type of order_tuple_in from List[Tuple(int)] -> List[List[int]]
        -Creates a list of monomials with specified exponents

        Parameters
        ----------
        variable_list : List[...]
            List of DReal variables which occur in the SMT equation
        order_tuple_in :List[tuple(int,.. ,int)]
            List of tuples which describe the exponent of each factor x_i^j, Here is i the position in the tuple
            j is the power of the factor. If the value is 0 then the factor is skipped

        Returns
        -------

        """

        order_list_list = [list(tuple_in) for tuple_in in order_tuple_in]
        dreal_monomials = [None] * len(order_list_list)

        def get_dreal_monomial_(j_in):
            """

            Parameters
            ----------
            j_in : int
                Position in monomial list
            """
            dreal_monomials[j_in] = np.prod(
                [variable_list[i] ** order_list_list[j_in][i] for i in range(len(order_list_list[0])) if
                 order_list_list[j_in][i] > 0])

        for j in range(len(order_list_list)):
            get_dreal_monomial_(j)

        return dreal_monomials

    @staticmethod
    def smt_abs(x):
        """
        Function which defines the absolute function for an SMT solver

        Parameters
        ----------
        x : ...
            Argument within the absolute function
        """

        return dr.if_then_else(x >= 0, x, -x)

    def smt_line_search(self, interval_constraints, error_func, left_bound, smt_tolerance, timeout=3.0,
                        max_iterations=10,
                        method='variable-stepsize'):
        """

        -First the configurations are set for the SMT problem.
        -Checks whether current error bound is satisfied or not
        -Converts the output flag is which is similar to z3
        -Update state + step based on the output flog

        Loop ends when there is an bound for which the SMT solver outputs unsatisfied AND the minimum amount of
        iterations has passed.


        Parameters
        ----------
        smt_tolerance
        timeout : float
            Amount of seconds the SMT solver is allowed to work on the problem per iteration
        interval_constraints
            SMT constraints which describe the domain
        error_func : ...
            The formula of the absolute difference between the exact and approximated Nth Lie derivative
        left_bound : float
            Minimum possible error of the bound
        max_iterations : int
            Minimum amount iterations that are required
        method
            step size tactic for the line search

        Returns
        -------

        """
        box = dr.Box(self.variable_list)
        unsat_found = -1
        config = dr.Config()
        config.use_polytope_in_forall = True
        config.use_polytope = True
        config.precision = smt_tolerance
        if method == 'variable-stepsize':
            xk = left_bound

            @kill_timeout.kill_timeout(timeout)
            def solve_dreal_problem():
                """

                Returns
                -------

                """

                return dr.CheckSatisfiability(smt_problem, config, box)

            step = 1
            i = 0
            while i <= max_iterations or unsat_found == -1:
                if i == 0:
                    i += 1
                    continue
                elif i == 1:
                    step = max(left_bound, 1)

                # Set the interval constraints first and append it with the polynomial at the end
                smt_problem = dr.And(*interval_constraints, self.smt_abs(error_func) >= xk + step)

                try:
                    res = solve_dreal_problem()

                except Exception as e:
                    print(e)
                    res = None
                # Do some multiprocessing

                # Convert unsatisfiable False to -1 to match with Z3
                # res = 0 when "Unknown"
                # res = 1 when "Satisfied"
                if res is False:
                    res = -1
                elif res is None:
                    res = 0

                xk, step, unsat_found = self.get_next_state(res, xk, step, unsat_found)
                i += 1
        return unsat_found

    @staticmethod
    def get_variable_list(n):
        """
        Returns the SMT variables of the problem

        Parameters
        ----------
        n : int
            Dimension of system
        """

        variable_list = [dr.Variable('x_%d' % i) for i in range(n)]
        return variable_list

class SMTErrorBound(object):
    """
    A class which is the base of classes which determine the maximum error between the Exact and approximated Nth Lie
    derivative

    ...

    Attributes
    ----------
        error : ndarray
            List of Upper bound of the difference of approximated and exact Nth Lie derivative
        interval_list : ndarray
            Description of the bounds of the domain
        lambda_list : ndarray
            Parameters of the Polyflow
        lie_sym_list : List[PolyMatrix]
            List of Lie derivatives saved as symbolic polynomials
        min_error : ndarray
            List of minimum error per entry
        n_dim : int
            Dimension of the system
        n_proj_matrices : int
            Amount of Lie derivatives - 1
        order_list : List[tuple]
            List tuples giving the power of each monomial
        projection_matrices_base : csr_matrix
            Matrices containing the coefficients of each monomial of each Lie derivatives
        projection_matrix : csr_matrix
            Matrix containing the coefficients of each monomial of the approximated Nth Lie derivative
        smt_equation_list
            Inequality equation to estimate the Nth Lie derivative. used for SMT solver
        spectral_radii : List[float]
            List of spectral_radii of each subsystem of the Polyflow
        sym_list : List[sympy.symbols]
            List of symbolic variables of the symbolic equations
        time_error : ndarray
            Diagonal matrix with "t * exp( rho_i * t)" on the diagonal
        time_step : float
            Time step in the reachability algorithm
        tolerance_error : List[float]
            User input tolerance on the Polyflow Error for the SMT solver
        tolerance_smt : List[float]
            Tolerance on the difference between the exact and approximated Nth Lie derivatives for the SMT solver.

    Methods
    ---------
    from_json
        Creates a SMTErrorBound object from a json variable
    to_json
        Converts the SMTErrorbound to a json string
    to_dict
        Converts the SMTErrorBound to a dictionary
    __init__
        Constructs the SMTErrorBound object
    get_lie_sym_list
        Returns the list of symbolic Lie derivatives
    get_lie_sym_list_json
        Returns the list of symbolic Lie derivatives as json string
    get_sym_list
        Returns a list of symbolics
    get_sym_list_json
        Returns list of symbolics as dictionary
    get_interval_list
        Returns the bounds of the interval
    get_interval_list_json
        Returns the bounds of the interval as dictionary
    get_min_error
        Get the minimal possible error for the SMT problem
    get_min_error_json
        Get the minimal possible error for the SMT problem
    set_lambda
        Set the Lambda parameters of the Polyflow
    get_lambda
        Returns the lambda parameters of the Polyflow
    get_lambda_json
        Returns the lambda parameters as a dictionary
    formulate_problems
        Formulates the problem to calculate the upperbound of the error between the approximated and exact
        Nth Lie derivative
    update_smt_coefficient_matrix
        Updates the coefficient matrix, is the difference between the exact and approximated Nth Lie derivative
    calculate_errors
        calculate the error of all entries of exact and approximated
    calculate_error
        calculate the error between exact and approximated
    smt_line_search
        Searches for an unsatisfied solution with the "lowest" bound
    get_next_state
        Handles the output of the SMT solver
    smt_abs
        defines the absolute function for the SMT functions
    get_smt_monomials
        Get all monomials, that can be used for the SMT solver
    """

    def to_json(self):
        """
        Returns a json string with all information to construct the SMTErrorBound object
        Returns
        -------
        String in json format
        """
        return json.dumps(self.to_dict())

    def to_dict(self):
        """
        Creates a dictionary with all attributes to construct an SMTErrorBound object
        Returns
        -------
        dictionary with all attributes
        """
        output_dict = {}
        output_dict.update(self.get_lie_sym_list_json())
        output_dict.update(self.get_sym_list_json())
        output_dict.update(self.get_interval_list_json())
        output_dict.update(self.get_min_error_json())
        return output_dict

    def __init__(self, lie_sym_list, sym_list,
                 interval_list, min_error,
                 scale_factor, time_step, spectral_radii, tolerance,
                 *args, **kwargs):
        """

        Parameters
        ----------
        lie_sym_list
            List of all Lie derivative up to order N
        sym_list
            List of all symbolics of the differential equation
        interval_list
            Description of the bounds of the domain
        min_error
            List of the lower bound of the error between exact and approximated solution
        scale_factor
            Scale factor used for coordinate transformation
        time_step
            time step of the reachability algorithm
        spectral_radii
            Spectral radius of each subsystem of the Polyflow
        tolerance
            User defined tolerance for the error
        args
        kwargs
        """

        # Set variables
        self.lambda_list = None
        self.smt_monoms = None
        self.time_error = np.ndarray
        n_dim = len(sym_list)
        n_lie = len(lie_sym_list)
        self.error = -1.0
        self.lie_sym_list = lie_sym_list
        self.sym_list = sym_list
        self.n_dim = n_dim
        self.n_proj_matrices = (n_lie - 1)
        self.projection_matrices_base = [csr_matrix] * n_lie
        self.interval_list = interval_list
        self.min_error = min_error

        # Get coefficients before the monomials which are used for  the SMT problem
        self.smt_equation_list = [None] * n_dim
        self.projection_matrix = csr_matrix((1, 1))
        for i in range(n_lie):
            self.projection_matrices_base[i], self.order_list = get_carleman_to_poly([lie_sym_list[i], ], sym_list)

        # Set constants
        self.scale_factor = scale_factor
        self.tolerance_error = tolerance
        self.time_step = time_step
        self.spectral_radii = spectral_radii

        # Get factor of the Polyflow error bound which depends on time and the spectral radius t * exp(t*rho)
        self.time_error = np.diag(self.get_time_error(scale_factor, time_step, spectral_radii, len(lie_sym_list) - 1))

        # Define the Tolerance for the SMT solver, which overapproximates the absolute difference between the Nth Lie
        # derivative and its approximation
        self.tolerance_smt = tolerance / self.time_error

        super(SMTErrorBound, self).__init__(*args, **kwargs)

    @staticmethod
    def get_time_error(scale_factor, time_step, spectral_radii, max_lie_order):
        """
        # Calculates the factor of the Polyflow error bound which depends on time and the spectral radius t * exp(t*rho)
        Parameters
        ----------
        scale_factor : float
            Scale factor used for the coordinate transformation
        time_step : float
            Time step of the reachability algorithm
        spectral_radii
            Spectral radius of the Polyflow subsystem
        max_lie_order
            Amount of observers used
        Returns
        -------

        """

        exponent_factor = np.diag(
            [np.exp(max(spectral_radii) * time_step) for _ in range(len(spectral_radii))])
        time_error = time_step * scale_factor ** (-max_lie_order + 1) * exponent_factor

        return time_error

    def get_lie_sym_list(self):
        """
        Get the symbolic functions of the Lie derivatives
        Returns
        -------
        self.lie_sym_list
        """
        return self.lie_sym_list

    def get_lie_sym_list_json(self):
        """
        Get the symbolic functions in dictionary format
        Returns
        -------

        """
        return {'lie_sym_list': self.get_lie_sym_list()}

    def get_sym_list(self):
        """
        Get the symbolics of the differential equation

        Returns
        -------

        """
        return self.sym_list

    def get_sym_list_json(self):
        """
        Get the symbolics of the differential equation in dictionary format
        Returns
        -------

        """
        return {'sym_list': self.get_sym_list()}

    def get_interval_list(self):
        """
        Get the bounds of the domain
        Returns
        -------

        """
        return self.interval_list

    def get_interval_list_json(self):
        """
        Get the bounds of the domain in dictionary format
        Returns
        -------

        """
        return {'interval_list': self.get_interval_list()}

    def get_min_error(self):
        """
        Get the lower bound of the upperbound
        Returns
        -------

        """
        return self.min_error

    def get_min_error_json(self):
        """
        Get the lower bound of the upperbound
        Returns
        -------

        """
        return {'min_error': self.min_error}

    def set_lambda(self, lambda_list):
        """
        Defines the problem to solve the Upper bound of the error between the exact and approximated
        Parameters
        ----------
        lambda_list
        """

        self.lambda_list = np.array(lambda_list)
        self.update_smt_coefficient_matrix()
        self.formulate_problems()

    def get_lambda(self):
        """
        Get the Polyflow parameters

        Returns
        -------

        """
        return self.lambda_list

    def get_lambda_json(self):
        """
        Get the Polyflow parameters in dictionary format
        Returns
        -------

        """
        return {'lambda_list': self.get_lambda().tolist()}

    def formulate_problems(self):
        """
        Formulate the equation which reflects the difference between the exact and approximated Nth Lie derivative
        Returns
        -------

        """

        self.smt_equation_list = get_smt_equation(self.smt_monoms, self.projection_matrix)

    def update_smt_coefficient_matrix(self):
        """
        Update the coefficient matrix used to formulate SMT problem
        Returns
        -------

        """
        # Allocate memory for coefficient matrix
        used_shape = self.projection_matrices_base[-1].shape
        self.projection_matrix = csr_matrix(used_shape)

        # Sum all Lie derivatives to N-1
        for i in range(self.n_proj_matrices):
            self.projection_matrix[:, :self.projection_matrices_base[i].shape[1]] = \
                self.projection_matrix[:, :self.projection_matrices_base[i].shape[1]] + \
                self.projection_matrices_base[i].multiply(self.lambda_list[:, i].reshape((-1, 1)))

        # Difference between the exact and estimated Nth Lie derivative
        self.projection_matrix[:, :] = \
            self.projection_matrix[:, :] \
            - self.projection_matrices_base[-1]

        return self.projection_matrix


class DrealErrorBound(DrealBase, SMTErrorBound):
    """
    A class used to derive the upperbound of the Nth Lie derivative using DReal

    ...

    Attributes
    ----------
    variable_list : Dreal variable
        Variable which represents the state variables of the problem x_0, x_1, etc.
    """

    def __init__(self, lie_sym_list, sym_list, interval_list, min_error,
                 scale_factor, time_step, spectral_radii, tolerance):
        """
        Constructor of the DRealErrorBound object
        Parameters
        ----------
        lie_sym_list : List[Polymatrix]
            List of Symbolic expressions which represents the Lie derivatives
        sym_list
            List of symbolics which represent the variables in the differential equation
        interval_list : ndarray
            Description of the Domain
        min_error
            Minimal value for the upperbound of the error
        """
        n_dim = len(sym_list)

        # Define Dreal variables
        self.variable_list = [dr.Variable('x_%d' % i) for i in range(n_dim)]

        super(DrealErrorBound, self).__init__(lie_sym_list=lie_sym_list, sym_list=sym_list, interval_list=interval_list,
                                              min_error=min_error, n=n_dim, scale_factor=scale_factor,
                                              time_step=time_step, spectral_radii=spectral_radii, tolerance=tolerance)

    def calculate_error(self, index, timeout=3000):
        """
        Overapproximates the maximum absolute difference between the exact and approximated Nth Lie derivative

        Parameters
        ----------
        index : int
            entry number
        timeout
            Maximum allowed time for the SMT solver
        Returns
        -------

        """

        dr_var = self.variable_list
        interval = self.interval_list

        n_interval = len(dr_var)

        interval_constraints = [np.ndarray] * n_interval * 2

        for i in range(n_interval):
            interval_constraints[2 * i] = dr_var[i] <= interval[i][1]
            interval_constraints[2 * i + 1] = interval[i][0] <= dr_var[i]

        min_error = self.min_error[index]
        smt_tolerance = self.tolerance_smt[index]
        return self.smt_line_search(interval_constraints, self.smt_equation_list[index], min_error, smt_tolerance)

class HigherOrderEstimationBase:
    """
    Class, which is used to estimate the Lagrangian remainder
    """

    def __init__(self, lie_fun, lie_sym, order, time_step, *args, **kwargs):
        """
        Constructs the object used to estimate the remainder of an m-1 th order Taylor series

        It multiplies the mth Lie derivative  with the Taylor coefficient 1/m! * time_step**m
        Constructs the monomials used in the Lie derivative with smt variables
        defines the smt equation of mth Lie derivative * Taylor coefficient

        Parameters
        ----------
        lie_fun : List[sympy]
            List with 1 element containing the mth Lie derivative
        lie_sym : sympy.symbols
            symbolics used in lie_fun
        order : int
            Indicates what term of the Taylor series is used to estimate the remainder
        time_step : float
            time step in the simulation
        args
        kwargs
        """

        taylor_coefficient = 1 / (math.factorial(order)) * time_step ** order
        lie_fun[0] *= taylor_coefficient

        self.dim = len(lie_sym)
        self.coefficient_mat, self.order_list = get_carleman_to_poly(lie_fun, lie_sym)
        self.order = order
        self.time_step = time_step
        self.variable_list = self.get_variable_list(self.dim)
        self.smt_monoms = self.get_smt_monomials(self.variable_list, self.order_list)

        self.define_function()

    def define_function(self):
        """
        Define the function for the lagrangian remainder. INCLUDING 1/i! * time_step**i
        Returns
        -------

        """
        raise NotImplementedError


class HigherOrderEstimationDreal(DrealBase, HigherOrderEstimationBase):
    """
     Class, which is used to estimate the Lagrangian remainder using DReal
    """

    def calculate_error(self, index, timeout=100):
        """

        Parameters
        ----------
        index
        timeout

        Returns
        -------

        """
        raise NotImplementedError

    def __init__(self, lie_fun, lie_sym, order, time_step, remainder_tolerance=0.01):

        super(HigherOrderEstimationDreal, self).__init__(lie_fun=lie_fun, lie_sym=lie_sym,
                                                         order=order, time_step=time_step, n=len(lie_sym), )
        self.problem_function = None
        self.config = dr.Config()
        self.config.use_polytope_in_forall = True
        self.config.use_polytope = True
        self.config.precision = remainder_tolerance
        self.box = dr.Box(self.variable_list)

    def define_function(self):
        """
        Defines the function of the lagrangian remainder
        Returns
        -------

        """
        self.problem_function = get_smt_equation(self.smt_monoms, self.coefficient_mat)

    # @timeit
    def calculate_remainder(self, interval_list, init_guess, timeout=3000):
        """
        Calculates the remainder
        Parameters
        ----------
        interval_list
        init_guess
        timeout

        Returns
        -------

        """

        @kill_timeout.kill_timeout(timeout)
        def solve_dreal_problem():
            """

            Returns
            -------

            """

            return dr.CheckSatisfiability(smt_problem, self.config, self.box)

        n = len(interval_list)
        interval_constraints = [None] * n
        dr_var = self.variable_list
        interval_center = np.mean(interval_list, axis=1)
        base_offset = abs(interval_list[:, 0] - interval_center)
        remainder_candidate = init_guess.__float__()
        while True:

            # Update interval
            for i in range(n):
                width = base_offset[i].__float__() + self.order*remainder_candidate
                interval_constraints[i] = width >= self.smt_abs(dr_var[i] - interval_center[i])

            valid_remainder = True
            # Check whether the remainder is outside the candidate domain
            for i in range(n):
                # .....
                smt_problem = dr.And(*interval_constraints,
                                     self.smt_abs(self.problem_function[i]) >= remainder_candidate)

                # break if res is unknown or satisfied
                try:
                    res = solve_dreal_problem()

                except Exception as e:
                    print(e)
                    res = None
                    # Time out exception only possible
                    pass

                if res:
                    valid_remainder = False

            if valid_remainder:
                break
            else:
                remainder_candidate *= 2

        return remainder_candidate
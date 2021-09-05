"""
    This file contains the functions related executing the PolyReach algorithm
"""

import copy
import time
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import sympy
from sympy import Poly
from sympy.polys.polymatrix import PolyMatrix
import json

import scripts.set_representations as pz
from scripts.polyflow import PolyFlow, Domain

from scripts.misc_functions import timeit_measure
from scripts.dreal_error_bound import HigherOrderEstimationDreal
from scripts.bernstein_bound import BernsteinBound


class PolyReach:
    """"
    PolyReach object used to solve reachability problems of polynomial systems
    ...

    Attributes
    ----------
    alpha_base_list : List[float]
        A list containing the the constant factors required for calculating the bloating factor
    dim_n : int
        Dimension of the differential equation
    domain : polyflow.Domain
        description of the domain of the reachability problem
    flow_pipe : set_representations.Zonotope
        Set representation which describes the trajectories of the interval [tk,tk+1]
    from_dict : Boolean
        Boolean whether the system is read from a json variable or not
    polyflow : polyflow.Polyflow
        Object containing the parameters + errorbound of Polyflow
    pz_lifted : set_representations.PZonotopeList
        Object representation the monomials of the lifted space at time step k
    pz_poly_flow_observer : set_representations.AugPZonotope
        Object representing the lifted state of polynomials (transformed coordinate system) at time step k
    pz_projected : set_representations.AugPZonotope
        Object representing the projected state at time step k + 1
    scale_factor : float
        Constant used for the coordinate system transformation
    time_step : float
        Difference in time between the time steps
    z_projected : set_representations.Zonotope
        Set representing the projected state at time step k + 1

    Methods
    ---------
    polyreach_from_dict(input_dict)
        Alternative constructor. Creates a PolyReach object with a dictionary
    to_dict
        Returns a dictionary of Polyreach object containing attribute information
    new_polyreach
        Creates a new PolyReach object
    init_alpha_base
        Creates a list of factors used to determine the bloating factors
    __str__
        Returns a string with information of Polyreach object
    define_dangerous_sets
        Defines the dangerous sets of the reachability problem
    simulate
        executes the reachability algorithm
    get_bloating_error
        Get the bloating error at time step k
    get_bloating_error2
        Get the bloating error at time step k
    create_flowpipe
        Create the set representation of all trajectories of the time interval [tk, tk+1]


    """

    bernstein_object: BernsteinBound
    remainder_obj: HigherOrderEstimationDreal
    time_step: float
    scale_factor: float
    remainder_smt_tol: List[float]
    pz_lifted: pz.PZonotopeList
    pz_projected: pz.AugPZonotope
    pz_poly_flow_observer: pz.AugPZonotope
    z_projected: pz.Zonotope
    dim_n: int
    domain: Domain
    flow_pipe: pz.Zonotope
    polyflow: PolyFlow
    alpha_base_list: List[float]
    extra_eig: float
    polyflow_smt_tol: float

    @timeit_measure
    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
            *param0 : MutablePolyDenseMatrix
                Differential equation of the system
            *param1 : tuple
                Tuple containing all symbolics used for the differential equation
            *param2 : list[list[float]
                2D list describing the grid for the Polyflow optimization
            *param3 : int
                Maximum Lie derivative that is estimated in the Polyflow
            *param4: float
                Time step of the simulation
            *param5: float
                relaxation factor for the Polyflow optimization
            *param6 : List[float]
                Relaxation term on the polyflow errorbound SMT problem (delta-weakening)
            *param : float
                Relaxation term on the Remainder SMT problem (delta-weakening)
        kwargs
            **name : str
                Name of the system
            **solver : str
                name of the solver used for the Polyflow optimization
            **smt_solver : str
                Name of the used SMT solver
            **plot : bool
                Determines whether a plot should be shown or not

            **polyflow : dict
                description of the polyflow object
            **flow_pipe : dict
                description of the allocated memory for the flowpipe per iteration
            **z_projected : dict
                description of the allocated memory for the over-approximated projected set per iteration
            **pz_projected
                description of the allocated memory for the approximated projected set per iteration
            **pz_lifted : dict
                description of the allocated memory for the lifted set (monomial) per iteration
            **pz_poly_flow_observer : dict
                description of the allocated memory for the lifted set (polynomial) per iteration
            **alpha_base_List : list
                list of factors used for the linear flowpipe
            **time_step : float
                time step of the simulation
            **dim_n : int
                Dimension of the system
            **from_dict : bool
                Boolean to decide whether new parameters have to be estimated or not
            **scale_factor : float
                Factor used for the coordinate transformation

        """
        self.dangerous_sets = []
        self.name = 'PolyReach'
        self.plot = False
        self.from_dict = False
        self.doi = None

        for key, value in kwargs.items():
            if key in ['name', 'plot']:
                setattr(self, key, value)

        if 'from_dict' in kwargs.keys():
            self.from_dict = kwargs['from_dict']
        if 'dangerous_sets' in kwargs.keys():
            self.dangerous_sets = [pz.Zonotope.from_interval_list(np.array(set_i))
                                   for set_i in kwargs['dangerous_sets']]

        if not self.from_dict:
            self.new_polyreach(*args, **kwargs)
        else:
            self.polyreach_from_dict(kwargs)

    def polyreach_from_dict(self, input_dict: dict):
        """
        Alternative constructor. Creates a PolyReach object with a dictionary
            This constructor transforms the type of the items in the dictionary to the correct, before it is set as an
            attribute of PolyReach.

        Parameters
        ----------
        input_dict
        Dictionary containing information to construct the PolyReach object.
        The Dictionary has the following keywords:
            [polyflow, flow_pipe, z_projected, pz_projected, pz_lifted
            pz_poly_flow_observer,alpha_base_list, time_step, dim_n, from_dict, scale_factor]

        Returns
        -------
        PolyReach
            returns the PolyReach with the entire problem description
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """ Returns a dictionary of Polyreach object containing attribute information """
        self.from_dict = True
        key_list = ['dim_n', 'polyflow', 'time_step', 'flow_pipe', 'z_projected',
                    'pz_projected', 'pz_poly_flow_observer', 'pz_lifted', 'name', 'remainder_smt_tol',
                    'alpha_base_list', 'from_dict', 'scale_factor']

        output_dict = {}
        for key_i in key_list:
            output_dict.update(to_json_el(self, key_i))

        return output_dict

    @staticmethod
    def _create_polyflow_object(extra_eig, polyflow_smt_tol, scale_factor, differential_eq,
                                symbol_tuple, domain_description, lie_order, time_step, **kwargs):
        # TODO Polyflow classmethod?

        input_dict_polyflow = copy.deepcopy(kwargs)
        input_dict_polyflow.update({'extra_eig': extra_eig,
                                    'polyflow_smt_tol': polyflow_smt_tol,
                                    'scale_factor': scale_factor})
        polyflow = PolyFlow(differential_eq, symbol_tuple, domain_description, lie_order,
                            time_step, **input_dict_polyflow)
        return polyflow

    @staticmethod
    def get_coordinate_scale_factor(time_step: float, lie_order: int, extra_eig: float) -> float:
        """ Calculates the scale factor used for the coordinate transformation """
        return (1 + extra_eig) ** -1 * (lie_order - 1) / time_step

    @staticmethod
    def _create_set_objects(dimension_projected, max_monomial_order, lie_order):
        # Allocate memory for big (polynomial) zonotope variables
        pz_lifted = pz.PZonotopeList.generate_list(dimension_projected, max_monomial_order)

        # Create empty polynomial zonotope to store the projected state. This shape is equal to the highest monomial
        generators_max_monomial = pz_lifted.polynomial_zonotope_list[-1].get_generators()
        e_max_monomial = pz_lifted.polynomial_zonotope_list[-1].get_e()
        projected_polynomial_zonotope_shape = generators_max_monomial.shape
        pz_projected = pz.AugPZonotope(np.empty((dimension_projected, 1)),
                                       None,
                                       np.empty((dimension_projected, projected_polynomial_zonotope_shape[-1])),
                                       e_max_monomial[:, 1:], is_empty=True)
        pz_poly_flow_observer = pz.AugPZonotope(np.empty((dimension_projected * lie_order, 1)),
                                                None,
                                                np.empty((dimension_projected * lie_order,
                                                          projected_polynomial_zonotope_shape[-1])),
                                                e_max_monomial[:, 1:], is_empty=True)

        # Allocate memory of zonotope which contains the projected state + polyflow error bound
        generators_temp = np.empty((dimension_projected, projected_polynomial_zonotope_shape[1] + dimension_projected))
        z_projected = pz.Zonotope(np.empty((dimension_projected, 1)),
                                  generators_temp, is_empty=True)

        # Allocate memory for flowpipe which exists of the following zonotopes
        flow_pipe = pz.Zonotope(np.empty((dimension_projected, 1)), np.empty((dimension_projected,
                                                                              generators_temp.shape[1] + 1
                                                                              + 2 * dimension_projected)),
                                is_empty=True)
        return pz_lifted, pz_projected, pz_poly_flow_observer, z_projected, flow_pipe

    def new_polyreach(self, differential_eq: PolyMatrix, symbol_tuple: Tuple[sympy.symbols],
                      domain_description: np.ndarray, lie_order: int, **kwargs):
        """

        Parameters
        ----------
        differential_eq
            Differential equation of the system
        symbol_tuple
            Symbolics used in the differential equation
        domain_description
            Description of the grid for the Polyflow optimization
        lie_order
            Maximum Lie order that is used for the optimization

        kwargs
            time_step
                Time step of the simulation
            extra_eig
                Relaxation factor for the polyflow optimization
            polyflow_smt_tol
                Relaxation factor for the SMT problem of the Polyflow errorbound (delta-weakening)
            remainder_smt_tol
                Relaxation factor for the SMT problem of the remainder (delta-weakening)
        Returns
        -------

        """

        prop_defaults = {
            'time_step': 0.1,
            'scale_factor': 1.0,
            'extra_eig': 0.2,
            'polyflow_smt_tol': None,
            'remainder_smt_tol': None,
            'doi': [0, 1]
        }

        # Set variables with default argument
        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))
            if prop in kwargs.keys():
                kwargs.pop(prop)

        order_remainder = 1

        self.dim_n = len(symbol_tuple)
        self.scale_factor = self.get_coordinate_scale_factor(self.time_step, lie_order, self.extra_eig)
        self.domain = Domain(domain_description)

        self.polyflow = self._create_polyflow_object(self.extra_eig, self.polyflow_smt_tol, self.scale_factor,
                                                     differential_eq, symbol_tuple, domain_description,
                                                     lie_order, self.time_step, **kwargs)

        if kwargs['smt_solver'] == 'dreal':
            self.remainder_obj = HigherOrderEstimationDreal([self.polyflow.lie_sympy_list[order_remainder], ],
                                                            self.polyflow.symbol_tuple, order_remainder, self.time_step,
                                                            self.remainder_smt_tol)
        # test
        self.bernstein_object = BernsteinBound(self.polyflow.lie_sympy_list[order_remainder],
                                               self.polyflow.symbol_tuple,
                                               order_remainder, self.time_step)

        self.pz_lifted, self.pz_projected, self.pz_poly_flow_observer, \
            self.z_projected, self.flow_pipe = self._create_set_objects(self.dim_n, self.polyflow.max_monomial_order,
                                                                        lie_order)

        # make it a polyflow function?
        self.init_alpha_base(self.polyflow.continuous_matrix_list)

    def init_alpha_base(self, matrices):
        """
        Creates a list of factors used to determine the bloating factors
            Initializes the constant used for the bloating factor of the flowpipe.
            This constant is only dependent on the eigenvalue of the operator
        Parameters
        ----------
        matrices
           Continuous time matrix of the system

        Returns
        -------
            list of floats
        """
        self.alpha_base_list = [-1] * self.dim_n
        for i in range(self.dim_n):
            self.alpha_base_list[i] = get_alpha_prime(matrices[i], self.time_step)

        return self.alpha_base_list

    def define_dangerous_sets_interval(self, interval_list):
        """ Defines the dangerous sets of the reachability problem """

        for interval_i in interval_list:
            self.dangerous_sets.append(pz.Zonotope.from_interval_list(interval_i))

        return self.dangerous_sets

    @timeit_measure
    def simulate(self, x_0: pz.Zonotope, t=10.0):
        """
        Solves the Reachability problem of PolyReach

        Parameters
        ----------
        x_0 : Zonotope
            Initial set
        t : float
            End time of simulation
        """
        time_measure_0 = time.time()
        time_step_i = 0

        snapshot_list = []
        flowpipe_list = []
        xk = x_0

        snapshot_list.append(xk)
        volume_list = np.empty(len(np.arange(0, t, self.time_step)))
        init_guess = 0.01
        for ti in np.arange(0, t, self.time_step):
            print('time %f' % ti)
            volume_list[time_step_i] = xk.get_volume()
            time_step_i += 1

            # Lift state
            self.pz_lifted.lift_n_to_m(xk)

            # Get bloating error
            self.pz_lifted.transform_to(self.polyflow.carl_to_poly_reduced, self.pz_poly_flow_observer)
            # bloating_error = self.get_bloating_error(self.pz_poly_flow_observer)

            # Project lifted state to next time step
            self.pz_lifted.transform_to(self.polyflow.operator, self.pz_projected)

            self.pz_projected.to_zonotope(self.z_projected, same_size=False)
            self.z_projected.GI[:, -self.dim_n:] = self.polyflow.polyflow_error

            # Create Flowpipe with xk and x_zono_projected as input
            # flowpipe_k = self.create_flowpipe(xk, self.z_projected, bloating_error)
            # flowpipe_k, init_guess = self.create_flowpipe_new(xk, self.remainder_obj, init_guess / 2)
            # flowpipe_k, init_guess = self.create_flowpipe_hoe(xk, self.z_projected, self.remainder_obj,
            # init_guess / 2)
            flowpipe_k, init_guess = \
                self.create_flowpipe_higher_order_estimation_bernstein(xk, self.z_projected,
                                                                       self.bernstein_object, init_guess / 2)

            # To order 1 the projected zonotope, so it can be used for next iteration
            xk = copy.deepcopy(pz.Zonotope.order_reduction_transform_method(self.z_projected))
            # xk = copy.deepcopy(pz.Zonotope.order_reduction_boxmethod(self.z_projected))
            # xk = copy.deepcopy(pz.Zonotope.order_reduction_pca(self.z_projected))
            snapshot_list.append(copy.deepcopy(xk))
            flowpipe_list.append(copy.deepcopy(flowpipe_k))

            is_safe = self._is_safe(xk, flowpipe_k, self.polyflow.domain_obj, self.dangerous_sets)

            if not is_safe:
                print(ti)
                break
        print('simulation time %f' % (time.time() - time_measure_0))
        self.dump_zonotopes(snapshot_list, 'xk_%s.json' % self.name)
        self.dump_zonotopes(flowpipe_list, 'flowpipe_%s.json' % self.name)
        xk_list = pz.ZonotopeList(snapshot_list, color='red', name='xi', doi=self.doi)
        flowpipe_list = pz.ZonotopeList(flowpipe_list, color='green', name='flowpipe', doi=self.doi)
        box_i = self.polyflow.domain_obj

        # self.plot_trajectory(xk_list, flowpipe_list, box_i)
        self.save_trajectory(xk_list, flowpipe_list, box_i)

        if self.plot:
            self.plot_trajectory(xk_list, flowpipe_list, box_i, self.dangerous_sets)

    @staticmethod
    def dump_zonotopes(zonotope_list, output_file):
        """ Writes list of zonotope objects to a json file """
        list_dict = [zono_i.to_dict() for zono_i in zonotope_list]
        with open(output_file, 'w') as f:
            json.dump(list_dict, f, indent=4)

    @staticmethod
    def _is_safe(_, flowpipe_k, domain, dangerous_sets):
        """ Determines whether simulation can continue or not """
        is_safe = flowpipe_k.is_in_domain(domain)
        for dangerous_i in dangerous_sets:
            if flowpipe_k.intersect_zonotope(dangerous_i):
                is_safe = False
                break
        return is_safe

    @staticmethod
    def create_flowpipe_higher_order_estimation_smt(x0, xk, remainder_obj, init_guess):
        """
        Creates a flowpipe using the 1st or 2nd derivative

        Parameters
        ----------
        xk
            Snap shot of system at time step k + 1
        x0
            Snap shot of system at time step k
        remainder_obj
            Object which is able to determine the remainder
        init_guess
            Initial guess of the remainder

        Returns
        -------
        Flowpipe of the system for [k, k+1]
        Remainder of system
        """

        ch = pz.Zonotope.over_approximate_2zonotope(x0, xk)
        ih_ch = ch.to_interval_hull()
        n = ch.get_set_dimension()

        lagrangian_remainder = remainder_obj.calculate_remainder(ih_ch.bounds, init_guess) * 2
        np.fill_diagonal(ih_ch.GI, ih_ch.GI.diagonal() + lagrangian_remainder)
        x_out = copy.deepcopy(ch)
        x_out += pz.Zonotope(np.zeros((n, 1)), np.eye(n) * lagrangian_remainder)
        return x_out, lagrangian_remainder

    @staticmethod
    def create_flowpipe_higher_order_estimation_bernstein(x0, xk, bernstein_object, init_guess):
        """
        Creates a flowpipe using the 1st or 2nd derivative

        Parameters
        ----------
        xk
            Snap shot of system at time step k + 1
        x0
            Snap shot of system at time step k
        bernstein_object
            Object which is able to determine the remainder
        init_guess
            Initial guess of the remainder

        Returns
        -------
        Flowpipe of the system for [k, k+1]
        Remainder of system
        """

        zonotope_overapprox = pz.Zonotope.over_approximate_2zonotope(x0, xk)
        interval_hull_overapprox = zonotope_overapprox.to_interval_hull()
        n = zonotope_overapprox.get_set_dimension()

        lagrangian_remainder = bernstein_object.calculate_remainder(interval_hull_overapprox.bounds, init_guess)
        np.fill_diagonal(interval_hull_overapprox.GI, interval_hull_overapprox.GI.diagonal() + lagrangian_remainder)
        x_out = copy.deepcopy(zonotope_overapprox)
        x_out += pz.Zonotope(np.zeros((n, 1)), np.eye(n) * lagrangian_remainder)
        return x_out, lagrangian_remainder

    @staticmethod
    def create_flowpipe_smt_zero_order(x0, remainder_obj, init_guess):
        """
        Creates a flowpipe using the 1st or 2nd derivative

        Parameters
        ----------
        x0
            Snap shot of system at time step k
        remainder_obj
            Object which is able to determine the remainder
        init_guess
            Initial guess of the remainder

        Returns
        -------
        Flowpipe of the system for [k, k+1]
        Remainder of system
        """

        # Get interval hull around initial set
        ih_0 = x0.to_interval_hull()
        n = x0.get_set_dimension()
        lagrangian_remainder = remainder_obj.calculate_remainder(ih_0.bounds, init_guess)
        np.fill_diagonal(ih_0.GI, ih_0.GI.diagonal() + lagrangian_remainder)
        x_out = copy.deepcopy(x0)
        x_out += pz.Zonotope(np.zeros((n, 1)), np.eye(n) * lagrangian_remainder)
        return x_out, lagrangian_remainder

    def save_trajectory(self, xk_list, flowpipe_list, domain, filename=None):
        """
        Saves list of sets to file which were converted to a vertex list

        Parameters
        ----------
        xk_list
            List of snap shots
        flowpipe_list
            List of flowpipes
        domain
            Polyflow domain
        filename
            Filename for output
        """
        if filename is None:
            filename = '%s_%f.json' % (self.name, time.time())

        output_dict = {}
        output_dict.update({'xk': xk_list.to_dict()})
        try:
            output_dict.update({'flowpipe': flowpipe_list.to_dict()})
        except Exception as e:
            print(e)
            print('Failed to get flowpipe saving it as None')
            output_dict.update({'flowpipe': None})
        output_dict.update(domain.to_dict())
        output_dict.update({'title': self.name})

        with open(filename, 'w') as f:
            json.dump(output_dict, f, indent=4)

    @staticmethod
    def plot_trajectory_from_file(input_file):
        """
        Plots trajectory from file

        Parameters
        ----------
        input_file
        Returns
        -------
        """
        with open(input_file) as json_file:
            input_dict = json.load(json_file)

        x_state = np.array(input_dict['xk']['vertices_list']['x'])
        y_state = np.array(input_dict['xk']['vertices_list']['y'])
        color_state = input_dict['xk']['color']
        state_dict = {'x_state': x_state,
                      'y_state': y_state,
                      'color_state': color_state}

        x_flowpipe = np.array(input_dict['flowpipe']['vertices_list']['x'])
        y_flowpipe = np.array(input_dict['flowpipe']['vertices_list']['y'])
        color_flowpipe = input_dict['flowpipe']['color']
        flowpipe_dict = {'x_flowpipe': x_flowpipe,
                         'y_flowpipe': y_flowpipe,
                         'color_flowpipe': color_flowpipe}

        domain = Domain(np.array(input_dict['domain']))
        box = domain.get_box(input_dict['xk']['doi'])
        doi = input_dict['xk']['doi']
        title = input_dict['title']
        plot_info_dict = {
            'domain': domain,
            'box': box,
            'doi': doi,
            'title': title
        }

        PolyReach.visualize(state_dict, flowpipe_dict, plot_info_dict)

    def plot_trajectory(self, xk_list, flowpipe_list, domain, dangerous_sets=None):
        """

        Parameters
        ----------
        xk_list
        flowpipe_list
        domain
        dangerous_sets

        Returns
        -------

        """
        if dangerous_sets is None:
            dangerous_sets = []
        x_state, y_state = xk_list.get_vertex_list()
        try:
            x_flowpipe, y_flowpipe = flowpipe_list.get_vertex_list()
        except Exception as e:
            print(e)
            print('Could not get vertices of flowpipe')
            x_flowpipe = np.zeros((4, 4))
            y_flowpipe = np.zeros((4, 4))
        box = domain.get_box(xk_list.doi)

        plot_info_dict = {
            'domain': domain,
            'box': box,
            'doi': xk_list.doi,
            'title': self.name
        }
        flowpipe_dict = {'x_flowpipe': x_flowpipe,
                         'y_flowpipe': y_flowpipe,
                         'color_flowpipe': flowpipe_list.color}
        state_dict = {'x_state': x_state,
                      'y_state': y_state,
                      'color_state': xk_list.color}

        self.visualize(state_dict, flowpipe_dict, plot_info_dict)

        for dangerous_i in dangerous_sets:
            plt.plot(*dangerous_i.interval_coordinate(self.doi), color='orange')

    @staticmethod
    def visualize(state_dict, flowpipe_dict, plot_info_dict):
        """
        TODO rename

        Parameters
        ----------
        state_dict
        flowpipe_dict
        plot_info_dict
        Returns
        -------
        """

        fig, ax = plt.subplots()
        ax.plot(flowpipe_dict['x_flowpipe'].T, flowpipe_dict['y_flowpipe'].T, flowpipe_dict['color_flowpipe'],
                alpha=0.5)
        ax.plot(state_dict['x_state'].T, state_dict['y_state'].T, state_dict['color_state'], alpha=0.5)
        ax.plot(plot_info_dict['box'][0], plot_info_dict['box'][1], 'k')

        plt.xlabel('$x_%d$' % (plot_info_dict['doi'][0] + 1))
        plt.ylabel('$x_%d$' % (plot_info_dict['doi'][1] + 1))
        plt.title(plot_info_dict['title'])
        ax.set_aspect('equal', 'box')
        plt.grid('on')

    def get_bloating_error(self, init_set):
        """
        Get the bloating error at time step k
            Here alpha_base_list is multiplied with the infinity norm of the set.
            After determining the bloating factor, the bloating factor is stored in a zonotope
        Parameters
        ----------
        init_set : set_representations.AugPZonotope
            Set in the lifted space
        Returns
        -------
        zonotope_out : set_representations.Zonotope
            Error box of the bloating error
        """
        bloating_error = np.eye(self.dim_n)
        x_inf = init_set.get_over_approx_inf()
        for i in range(self.dim_n):
            bloating_error[i, i] = self.alpha_base_list[i] * x_inf

        zonotope_out = pz.Zonotope(np.zeros((self.dim_n, 1)), bloating_error)

        return zonotope_out

    def get_bloating_error2(self, init_set):
        """
        Get the bloating error at time step k

        Parameters
        ----------
        init_set

        Returns
        -------

        """
        bloating_error = np.eye(self.dim_n)
        x_inf = init_set.get_over_approx_inf()
        spectral_radii = self.polyflow.get_2norms(self.polyflow.continuous_matrix_list)
        for i in range(self.dim_n):
            bloating_error[i, i] = self.time_step ** 2 * np.exp(spectral_radii[i] * self.time_step) * spectral_radii[
                i] ** 2 * x_inf

        zonotope_out = pz.Zonotope(np.zeros((self.dim_n, 1)), bloating_error)

        return zonotope_out

    @staticmethod
    def create_flowpipe_linear(zonotope_start, zonotope_end, bloating_error):
        """
        Create the set representation of all trajectories of the time interval [tk, tk+1]
            The following steps are done
            -Calculating the "convex hull" of xk and xk+1
            -Adding the bloating term to the "convex hull"
        Parameters
        ----------
        zonotope_start : set_representations.Zonotope
            Set of the state at time step tk
        zonotope_end
            Set of the state at time step tk+1
        bloating_error
            Term to over-approximate the trajectories of the interval [tk, tk+1]

        Returns
        -------
        zonotope_flowpipe : set_representations.Zonotope
            Set containing all trajectories of the interval [tk, tk+1]
        """

        zonotope_flowpipe = pz.Zonotope.over_approximate_2zonotope(zonotope_start, zonotope_end)
        zonotope_flowpipe += bloating_error

        return zonotope_flowpipe

    def write_back_up(self, filename):
        """

        Parameters
        ----------
        filename

        Returns
        -------

        """
        with open(filename[:-5] + '_backup.json', 'w') as f:
            json.dump(self.to_dict(), f, indent=4)


def get_alpha_prime(matrix_in, time_step):
    """
    Calculates the constant factor which is used to calculate the bloating term
        The factor depends on the time step and the spectral radius of the continuous time matrix
        -First the spectral radius is calculated.
        -The factor is calculated with the formula of ...
    Parameters
    ----------
    matrix_in : ndarray
        Matrix of the continuous time system
    time_step : float
        Time step of the simulation
    Returns
    -------
    alpha_prime :float
        Factor used to overapproximate the trajectories
    """

    # Calculate the spectral radius
    spectral_radius = np.max(np.abs(np.linalg.svd(matrix_in, compute_uv=False)))

    # Calculate the constant factor to overapproximate all trajectories
    alpha_prime = np.exp(time_step * spectral_radius) - 1 - time_step * spectral_radius
    return alpha_prime


def read_parameter_file(file_name):
    """

    Parameters
    ----------
    file_name : str
        file location of parameter file

    Returns
    -------

    """
    with open(file_name) as json_file:
        input_dict = json.load(json_file)

    te = float(input_dict['te'])
    initial_set_list = np.array(input_dict['X0'])
    if len(initial_set_list.shape) == 2:

        initial_set_list = [pz.Zonotope.from_interval_list(initial_set_list), ]
    elif len(initial_set_list.shape) == 3:
        initial_set_list = [pz.Zonotope.from_interval_list(initial_set_list[:, :, i]) for i in range(initial_set_list)]
    else:
        raise ValueError

    # If there is a file with the parameters available
    if 'model_file' in input_dict.keys():
        with open(input_dict['model_file']) as model_file:
            model_dict = json.load(model_file)

        if 'plot' in input_dict.keys():
            model_dict.update({'plot': input_dict['plot']})
        if 'name' in input_dict.keys():
            model_dict.update({'name': input_dict['name']})

        p_obj = PolyReach(**model_dict)

    else:

        domain_description, symbol_tuple, diff_function, delta_t, \
            lie_order, polyflow_smt_tol, remainder_smt_tol = parse_file_dictionary(input_dict)
        second_dict = _parse_auxiliary_variables(input_dict)

        p_obj = PolyReach(diff_function, symbol_tuple, domain_description, lie_order, time_step=delta_t, extra_eig=0.1,
                          polyflow_smt_tol=polyflow_smt_tol,
                          remainder_smt_tol=remainder_smt_tol, **second_dict)

        with open(file_name[:-5] + '_backup.json', 'w') as f:
            json.dump(p_obj.to_dict(), f, indent=4)

    for x0 in initial_set_list:
        p_obj.simulate(x0, te)


def parse_file_dictionary(input_dict):
    """

    Parameters
    ----------
    input_dict

    Returns
    -------

    """
    domain_description = np.array(input_dict['domain'])
    symbol_tuple = tuple(sympy.sympify(input_dict['state_var']))
    diff_function = PolyMatrix([Poly(diff_i, symbol_tuple) for diff_i in sympy.sympify(input_dict['diff_eq'])])

    delta_t = float(input_dict['delta_t'])
    lie_order = int(input_dict['lie_order'])
    polyflow_smt_tol = np.array(input_dict['polyflow_smt_tol'])
    remainder_smt_tol = input_dict['remainder_smt_tol']

    return domain_description, symbol_tuple, diff_function, delta_t, \
           lie_order, polyflow_smt_tol, remainder_smt_tol


def to_json_el(self_variable, key_name):
    """ In this function the attribute is converted if necessary to a json friendly variable """
    try:
        attr = getattr(self_variable, key_name)
    except Exception as e:
        print('key name: %s has failed' % key_name)
        print(e)
        return
    if type(attr) in [pz.Zonotope, pz.ZonotopeList, pz.AugPZonotope,
                      pz.PZonotopeList, pz.Domain, PolyFlow]:
        attr = attr.to_dict()
    else:
        pass

    return {key_name: attr}


def _parse_auxiliary_variables(input_dict):
    """ Parse the optional arguments for PolyReach """
    second_dict = {}
    for key_name in ['name', 'solver', 'smt_solver', 'plot', 'doi', 'dangerous_sets']:
        if key_name in input_dict.keys():
            second_dict.update({key_name: input_dict[key_name]})
    return second_dict

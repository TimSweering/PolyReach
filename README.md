# PolyReach

**Author:** [T.H.J. Sweering](https://www.linkedin.com/in/tim-sweering-1b5418190/)

## About
This repository contains the reachability tool PolyReach. 
The reachability tool focuses on polynomial systems within a predefined domain.

This reachability tool uses the Polyflow method [1] to determine its next state. 
The method is a semi linear approach since it has to compute nonlinear functions and uses a linear matrix to map its state to the next time step.

The tool has two phases: the identification phase and the simulation phase. The simulation phase is similar to other nonlinear reachability algorithms. 
However, this tool uses a constant value for the bound on the error of the approximation of the state.
The identification phase is used to precompute parameters and find the corresponding error bound. 

The tool shifts a part of the computational work of the reachability problem to the identification phase, which is beneficial when multiple trajectories have to be simulated.

For the set representation the tool uses zonotopes and polynomial zonotopes.
Common operations on these sets are described in [2] and [3].

This tool provides two approaches for the flowpipe overapproximation. The linear approach as explained in [4] and the nonlinear approach of [5].
The linear approach is computationally less complex, but it is more conservative than the nonlinear approach. 





## Installation

This tool has run successfully on Ubuntu 20.04 and uses Python 3.8.10.

This tool uses the SMT-Solver [DReal](https://github.com/dreal/dreal4).

For the optimization part we use [CVXPY](https://github.com/cvxpy/cvxpy).  One can choose between multiple algorithms which are suitable to solve LP problems with numerous linear inequality constraints: CPLEX, SCS and Gurobi.

## Dependencies

<ul>
<li>aesara==2.0.10</li>
<li>cloudpickle==1.6.0</li>
<li>colorama==0.4.4</li>
<li>commonmark==0.9.1</li>
<li>cplex==20.1.0.1</li>
<li>cvxpy==1.1.13</li>
<li>cycler==0.10.0</li>
<li>dreal==4.21.6.1</li>
<li>ecos==2.0.7.post1</li>
<li>filelock==3.0.12</li>
<li>gurobipy==9.1.2</li>
<li>jsonlib-python3==1.6.1</li>
<li>kill-timeout==0.0.3</li>
<li>kiwisolver==1.3.1</li>
<li>llvmlite==0.36.0</li>
<li>matplotlib==3.4.2</li>
<li>memory-profiler==0.58.0</li>
<li>mpmath==1.2.1</li>
<li>numba==0.53.1</li>
<li>numpy==1.20.3</li>
<li>nvidia-ml-py==11.450.51</li>
<li>osqp==0.6.2.post0</li>
<li>Pillow==8.2.0</li>
<li>psutil==5.8.0</li>
<li>Pygments==2.9.0</li>
<li>pyparsing==2.4.7</li>
<li>pypolycontain==1.4</li>
<li>python-dateutil==2.8.1</li>
<li>qdldl==0.1.5.post0</li>
<li>rich==10.5.0</li>
<li>scalene==1.3.8</li>
<li>scikit-glpk==0.4.1</li>
<li>scipy==1.6.3</li>
<li>scs==2.1.4</li>
<li>six==1.16.0</li>
<li>sympy==1.8</li>
<li>tblib==1.7.0</li>
<li>yappi==1.3.2</li>
</ul> 


### Ubuntu 20.04
  

To install all required python modules use
> pip install -r requirements.txt




## Getting Started
The PolyReach tool uses *.json files to store its parameters. There are 3 types of parameter files for the tool. 
<ul>
<li>System</li>
<li>Estimated parameters</li>
<li>Flowpipe + overapproximations of discrete states</li>
</ul> 

### Analysing a System

To analyse a system use the following command, where the parameter file should contain the variables described below. 
> ./PolyReach.py <parameter_file>.json

It outputs whether or not the system is safe and additionally saves two files: one with the estimated parameters and one with the flowpipe and the overapproximations of the discrete states. 

Name variable | Type | Info
--- | --- | ---
name | `String` | Name of system 
plot | `Bool` | Whether the plot is shown after simulation
domain | `List[List[Float]]` | Description of grid for the Polyflow optimization. Size of list is n x 3
state_var | `String` | variables in lexicographic order and `,` as seperator
diff_eq | `String` | All differential equations in sympy syntax 
delta_t | `Float` | Time step of the simulation
doi | `List[Int]` | Axes which are shown in the plot (default [0, 1])
te | `Float` | End time of the simulation
lie_order | `Int` | Lie order which is estimated in the Polyflow
X0 | `List[List[Float]]` | Initial set of the simulation. The list is an (n x 2) list describing an n-dimensional interval
smt_solver | `String` | Chosen SMT solver: `dreal` (default)
solver | `String` | Chosen optimization algorithm for Polyflow optimization :`CPLEX`/`SCS`/`GUROBI`
polyflow_smt_tol | `List[Float]` |  Relaxation factor used for the Polyflow errorbound estimation
remainder_smt_tol | `Float`  | Relaxation factor used for the remainder estimation (default 0.01)
dangerous_sets | `List[List[List[Float]]]`  | List of dangerous sets (represented by multidimensional intervals)


### Example 

In the following an example of such a parameter file is given. This parameter file is used in one of experiments of PolyReach [6].

```json
{
    "name" : "Van der Pool oscillator with collision",
    "plot" : false,
    "domain": [
        [-5, 5, 1],
        [-5, 5, 1]
    ],
    "state_var": "x0, x1",
    "diff_eq": "[x1, 0.5*(1-x0**2)*x1-x0]",
    "delta_t": 0.005,
    "te": 7,
    "lie_order": 8,
    "X0": [
        [1.25, 1.55],
        [2.25, 2.35]
    ],
    "smt_solver" : "dreal",
    "solver" : "GUROBI",
    "polyflow_smt_tol" : [5E-6, 12E-5],
    "remainder_smt_tol" : 0.01,
    "dangerous_sets" : [
        [
            [-3,-1],
            [-1,1]
        ],
        [
            [-3,-1],
            [1.5,2.5]
        ]
    ]
}
```


### Plot Trajectory
After simulating a system a `.json` file containing all trajectories has been output.
To plot the trajectory use the following command
> ./PlotPolyreach <parameter_file>.json

## References

[1] Polyflow source
R. M. Jungers and P. Tabuada, “Non-local linearization of nonlinear differential equations via polyflows,” in 2019 American Control Conference (ACC), pp. 1–6, IEEE, 2019

[2] N. Kochdumper and M. Althoff, “Sparse polynomial zonotopes: A novel set representation for reachability analysis,” IEEE Transactions on Automatic Control, 2020.

[3]  A.-K. Kopetzki, B. Schürmann, and M. Althoff, “Methods for order reduction of zonotopes,” in 2017 IEEE 56th Annual Conference on Decision and Control (CDC), pp. 5626–5633, IEEE, 2017.

[4] A. Girard, "Reachability of uncertain linear systems using zonotopes." International Workshop on Hybrid Systems: Computation and Control. Springer, Berlin, Heidelberg, 2005.

[5] T. Dreossi, T. Dang, and C. Piazza, "Reachability computation for polynomial dynamical systems." Formal Methods in System Design 50.1 (2017): 1-38.

[6] T.H.J. Sweering, "Applying Koopman Methods for Nonlinear Reachability Analysis," 2021. 
#!/usr/bin/env python3
# 

"""
Solve Poisson equation by Finite Element methods

.. math::
    -\Delta u = f

with the weak formulation

.. math::
    (\\nabla u, \\nabla v) = (f, v)

Notes
-----

"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import MeshFactory as MF

# Try to implement the space class named `BernsteinFiniteElementSpace`

# from BernsteinFiniteElementSpace import BernsteinFiniteElementSpace

from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve



parser = argparse.ArgumentParser(description=
        """
        Lagrange finite element methods on triangulation.
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='the degree of the space, default value is 1.')

parser.add_argument('--dim',
        default=2, type=int,
        help='the dimension of the pde model, default value is 2.')

parser.add_argument('--ns',
        default=10, type=int,
        help='the number of segments on each axis, default value is 10.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='the number of iterations, default value is 4')

args = parser.parse_args()

degree = args.degree
dim = args.dim
ns = args.ns
maxit = args.maxit

if dim == 2:
    from fealpy.pde.poisson_2d import CosCosData as PDE
    domain = [0, 1, 0, 1]
    mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
elif dim == 3:
    from fealpy.pde.poisson_3d import CosCosCosData as PDE
    domain = [0, 1, 0, 1, 0, 1]
    mesh = MF.boxmesh3d(domain, nx=ns, ny=ns, nz=ns, meshtype='tet')

pde = PDE()

errorType = ['$|| u - u_h||_{\Omega,0}$',
             '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$'
             ]
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)

for i in range(maxit):
    print("The {}-th computation:".format(i))

    # space = BernsteinFiniteElementSpace(mesh, p=degree)
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    NDof[i] = space.number_of_global_dofs()
    bc = DirichletBC(space, pde.dirichlet) 

    uh = space.function()
    A = space.stiff_matrix()
    F = space.source_vector(pde.source)
    A, F = bc.apply(A, F, uh)
    uh[:] = spsolve(A, F).reshape(-1)

    errorMatrix[0, i] = space.integralalg.error(pde.solution, uh)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, uh.grad_value)

    if i < maxit-1:
        mesh.uniform_refine()


if dim == 2:
    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1, projection='3d')
    uh.add_plot(axes, cmap='rainbow')
elif dim == 3:
    print('The 3d function plot is not been implemented!')

showmultirate(plt, 0, NDof, errorMatrix,  errorType, 
        propsize=40)

plt.show()

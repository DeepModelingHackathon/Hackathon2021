
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace

"""
`NC`: the number of cells in mesh
`TD`: the topological dimension of mesh
`NQ`: represents the number of integral points 
"""

# the degree of Lagrange finite element space
p = 1 
 
# the domain [0, 1]^3
domain = [0, 1, 0, 1, 0, 1]

# generate the tetrahedron mesh 
mesh = MF.boxmesh3d(domain, nx=10, ny=10, nz=10, meshtype='tet')
NC = mesh.number_of_cells() # NC = 6*10*10*10

# the continuous piecewise finite element space of degree p
# spacetype='C' means continuous
space = LagrangeFiniteElementSpace(mesh, p, spacetype='C')


# get q-th integral formula
q = p+2
qf = mesh.integrator(q, 'cell') 
# The integral point in barycentric form and bcs.shape = (NQ, TD+1) 
# The integral weights and ws.shape == (NQ, )
bcs, ws = qf.get_quadrature_points_and_weights() 

# the cell measure array with `shape=(NC, )`
cellmeasure = mesh.entity_measure('cell') 

# （NQ, NC, ldof, GD)
gphi = space.grad_basis(bcs)

# construct the cell stiff matrix ， A.shape == (NC, ldof, ldof)
A = np.einsum('i, ijkl, ijml, j->jkm', ws, gphi, gphi, cellmeasure)

# The global number array of each local degree of freedom on the each cell 
cell2dof = space.cell_to_dof()

# (NC, ldof) --> (NC, ldof, 1) --> (NC, ldof, ldof)
I = np.broadcast_to(cell2dof[:, :, None], shape=A.shape)

# (NC, lodf) --> (NC, 1, ldof) --> (NC, ldof, ldof)
J = np.broadcast_to(cell2dof[:, None, :], shape=A.shape)


# the total number of global degree of freedoms 
gdof = space.number_of_global_dofs() 

# The global stiff matrix
A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))

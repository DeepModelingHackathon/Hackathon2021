import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve


from fealpy.functionspace.Function import Function

from fealpy.functionspace.femdof import multi_index_matrix1d
from fealpy.functionspace.femdof import multi_index_matrix2d
from fealpy.functionspace.femdof import multi_index_matrix3d

from fealpy.functionspace.femdof import multi_index_matrix

from fealpy.functionspace.femdof import CPLFEMDof1d, CPLFEMDof2d, CPLFEMDof3d
from fealpy.functionspace.femdof import DPLFEMDof1d, DPLFEMDof2d, DPLFEMDof3d

from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.decorator import timer, barycentric


class BernsteinFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C', q=None, dof=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.p = p
        if dof is None:
            if spacetype == 'C':
                if mesh.meshtype == 'interval':
                    self.dof = CPLFEMDof1d(mesh, p)
                    self.TD = 1
                elif mesh.meshtype == 'tri':
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'halfedge2d':
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'stri':
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'tet':
                    self.dof = CPLFEMDof3d(mesh, p)
                    self.TD = 3
            elif spacetype == 'D':
                if mesh.meshtype == 'interval':
                    self.dof = DPLFEMDof1d(mesh, p)
                    self.TD = 1
                elif mesh.meshtype == 'tri':
                    self.dof = DPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'tet':
                    self.dof = DPLFEMDof3d(mesh, p)
                    self.TD = 3
        else:
            self.dof = dof
            self.TD = mesh.top_dimension() 

        if len(mesh.node.shape) == 1:
            self.GD = 1
        else:
            self.GD = mesh.node.shape[1]

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.multi_index_matrix = multi_index_matrix 
        self.stype = 'lagrange'

    def __str__(self):
        return "Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell2dof[index]

    def face_to_dof(self, index=np.s_[:]):
        return self.dof.face_to_dof()

    def edge_to_dof(self, index=np.s_[:]):
        return self.dof.edge_to_dof()

    def boundary_dof(self, threshold=None):
        if self.spacetype == 'C':
            return self.dof.boundary_dof(threshold=threshold)
        else:
            raise ValueError('This space is a discontinuous space!')

    def is_boundary_dof(self, threshold=None):
        if self.spacetype == 'C':
            return self.dof.is_boundary_dof(threshold=threshold)
        else:
            raise ValueError('This space is a discontinuous space!')

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

 
    @barycentric
    def basis(self, bc, index=np.s_[:], p=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(1, 1, ldof)` or `(NQ, 1, ldof)`

        See Also
        --------

        Notes
        -----

        """
        pass
        
    @barycentric
    def grad_basis(self, bc, index=np.s_[:], p=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`

        Returns
        -------
        gphi : numpy.ndarray
            the shape of `gphi` can b `(NC, ldof, GD)' or
            `(NQ, NC, ldof, GD)'

        See also
        --------

        Notes
        -----

        """
        pass

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        TD = bc.shape[-1] - 1
        phi = self.basis(bc)
        e2d = self.dof.entity_to_dof(etype=TD, index=index)

        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[e2d])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        """
        Notes
        -----
        不同维度的实体
        """
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, gphi, uh[cell2dof[index]])
        return val

    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        dim = len(uh.shape)
        GD = self.geo_dimension()
        if (dim == 2) & (uh.shape[1] == GD):
            val = self.grad_value(uh, bc, index=index)
            return val.trace(axis1=-2, axis2=-1)
        else:
            raise ValueError("The shape of uh should be (gdof, gdim)!")

    def projection(self, u):
        """
        Project a function u into this space
        """
        M= self.mass_matrix()
        F = self.source_vector(u)
        uh = self.function()
        uh[:] = spsolve(M, F).reshape(-1)
        return uh

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array, coordtype='barycentric')
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)

    def integral_basis(self):
        """
        """
        cell2dof = self.cell_to_dof()
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs)
        cc = np.einsum('m, mik, i->ik', ws, phi, self.cellmeasure)
        gdof = self.number_of_global_dofs()
        c = np.zeros(gdof, dtype=self.ftype)
        np.add.at(c, cell2dof, cc)
        return c


    def stiff_matrix(self, c=None, q=None, isDDof=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.grad_basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, c=c, q=q)

        if isDDof is not None: # 处理 D 氏边界条件
            bdIdx = np.zeros(A.shape[0], dtype=np.int_)
            bdIdx[isDDof] = 1
            Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
            T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
            A = T@A@T + Tbd
        return A 

    def mass_matrix(self, c=None, q=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, c=c, q=q)
        return A 

    def source_vector(self, f, dim=None, q=None):
        p = self.p
        cellmeasure = self.cellmeasure
        bcs, ws = self.integrator.get_quadrature_points_and_weights()

        if f.coordtype == 'cartesian':
            pp = self.mesh.bc_to_point(bcs)
            fval = f(pp)
        elif f.coordtype == 'barycentric':
            fval = f(bcs)

        gdof = self.number_of_global_dofs()
        shape = gdof if dim is None else (gdof, dim)
        b = np.zeros(shape, dtype=self.ftype)

        if p > 0:
            if type(fval) in {float, int}:
                if fval == 0.0:
                    return b
                else:
                    phi = self.basis(bcs)
                    bb = np.einsum('m, mik, i->ik...', 
                            ws, phi, self.cellmeasure)
                    bb *= fval
            else:
                phi = self.basis(bcs)
                bb = np.einsum('m, mi..., mik, i->ik...',
                        ws, fval, phi, self.cellmeasure)
            cell2dof = self.cell_to_dof() #(NC, ldof)
            if dim is None:
                np.add.at(b, cell2dof, bb)
            else:
                np.add.at(b, (cell2dof, np.s_[:]), bb)
        else:
            b = np.einsum('i, ik..., k->k...', ws, fval, cellmeasure)

        return b

    def set_dirichlet_bc(self, uh, gD, threshold=None, q=None):
    	"""
        Set the Dirichlet boundary condition into uh.

        Notice that the Bernstein finite element space don't satisfy the
        interpolation property.

    	"""
 	pass

    def set_neumann_bc(self, F, gN, threshold=None, q=None):

        """

        Notes
        -----
        设置 Neumann 边界条件到载荷向量 F 中

        TODO: 考虑更多 gN 的情况, 比如 gN 可以是一个数组
        """
        p = self.p
        mesh = self.mesh

        dim = 1 if len(F.shape)==1 else F.shape[1]
       
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof()[index]
        n = mesh.face_unit_normal(index=index)
        measure = mesh.entity_measure('face', index=index)

        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = self.face_basis(bcs)

        pp = mesh.bc_to_point(bcs, index=index)
        val = gN(pp, n) # (NQ, NF, ...), 这里假设 gN 是一个函数

        bb = np.einsum('m, mi..., mik, i->ik...', ws, val, phi, measure)
        if dim == 1:
            np.add.at(F, face2dof, bb)
        else:
            np.add.at(F, (face2dof, np.s_[:]), bb)

    def set_robin_bc(self, A, F, gR, threshold=None, q=None):
        """

        Notes
        -----

        设置 Robin 边界条件到离散系统 Ax = b 中.

        TODO: 考虑更多的 gR 的情况

        """
        p = self.p
        mesh = self.mesh
        dim = 1 if len(F.shape) == 1 else F.shape[1]

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof()[index]

        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        measure = mesh.entity_measure('face', index=index)

        phi = self.face_basis(bcs)
        pp = mesh.bc_to_point(bcs, index=index)
        n = mesh.face_unit_normal(index=index)

        val, kappa = gR(pp, n) # (NQ, NF, ...)

        bb = np.einsum('m, mi..., mik, i->ik...', ws, val, phi, measure)
        if dim == 1:
            np.add.at(F, face2dof, bb)
        else:
            np.add.at(F, (face2dof, np.s_[:]), bb)

        FM = np.einsum('m, mi, mij, mik, i->ijk', ws, kappa, phi, phi, measure)

        I = np.broadcast_to(face2dof[:, :, None], shape=FM.shape)
        J = np.broadcast_to(face2dof[:, None, :], shape=FM.shape)

        A += csr_matrix((FM.flat, (I.flat, J.flat)), shape=A.shape)

        return A, F


    def to_function(self, data):
        p = self.p
        if p == 1:
            uh = self.function(array=data)
            return uh
        elif p == 2:
            cell2dof = self.cell_to_dof()
            uh = self.function()
            uh[cell2dof] = data[:, [0, 5, 4, 1, 3, 2]]
            return uh



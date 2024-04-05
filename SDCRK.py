import numpy as np
from qmatrix import genCollocation, genQDelta
from vectorize import matVecInv, matVecMul
from lagrange import LagrangeApproximation


class RKAnalysis(object):

    def __init__(self, A, b, c):

        A = np.asarray(A)      # shape (s, s)
        b = np.asarray(b)      # shape (s,)
        c = np.asarray(c)      # shape (s,)

        # Check A dimension
        assert A.shape[0] == A.shape[1], \
            f"A {A.shape} is not a square matrix"
        self.A = A

        # Check b and c dimensions
        assert b.size == self.s, \
            f"b (size={b.size}) has not correct size : {self.s}"
        assert c.size == self.s, \
            f"c (size={c.size}) has not correct size : {self.s}"
        self.b, self.c = b, c

        # Utility matrices
        self.eye = np.eye(self.s)      # shape (s, s)
        self.ones = np.ones(self.s)    # shape (s,)

        # Storage for order
        self._order = None


    @property
    def s(self):
        """Number of stages"""
        return self.A.shape[0]


    def printTableau(self, digit=4):
        """
        Print the Butcher table

        Parameters
        ----------
        digit : int, optional
            Number of digit for each coefficients. The default is 4.
        """
        print("Butcher table :")
        acc = f" .{digit}f"
        bi = ""
        for i in range(self.s):
            bi += f'{float(self.b[i]):{acc}} '
            Ai = ""
            for j in range(self.s):
                Ai += f'{float(self.A[i,j]):{acc}} '
            line = f'{float(self.c[i]):{acc}} | {Ai}'
            if i == 0:
                print('-'*(len(line)-1))
            print(line)
        print('-'*(len(line)-1))
        print(f"{(digit+3)*' '} | {bi}")
        print('-'*(len(line)-1))


    def stabilityFunction(self, z):
        """
        Compute the stability function of the RK method

        Parameters
        ----------
        z : scalar or vector or matrix or ...
            Complex values where to evaluate the stability function.

        Returns
        -------
        sf : scalar or vector or matrix or ...
            Stability function evaluated on the complex values.
        """
        # Pre-processing for z values
        z = np.asarray(z)
        shape = z.shape
        z = z.ravel()[:, None, None]    # shape (nValues, 1, 1)

        print(f"Computing Butcher stability function for {z.size} values")

        # Prepare Butcher tables
        A = self.A[None, :, :]          # shape (1, s, s)
        b = self.b[None, None, :]       # shape (1, 1, s)
        eye = self.eye[None, :, :]      # shape (1, s, s)
        ones = self.ones[:, None]       # shape (s, 1)

        # Compute 1 + z*b @ (I-zA)^{-1} @ ones
        sf = 1 + (z*b) @ matVecInv(eye - z*A, ones)

        # Reshape to original size (scalar or vector)
        sf.shape = shape

        return sf
  

    @property
    def orderFromPython(self):
        conditions = {
            1: [('i', 1)],
            2: [('i,ij', 1/2)],
            3: [('i,ij,ik', 1/3),
                ('i,ij,jk', 1/6)],
            4: [('i,ij,ik,il', 1/4),
                ('i,ij,jl,ik', 1/8),
                ('i,ij,jk,jl', 1/12),
                ('i,ij,jk,kl', 1/24)],
            5: [('i,ij,ik,il,im', 1/5),
                ('i,ij,jk,il,im', 1/10),
                ('i,ij,jk,jl,im', 1/15),
                ('i,ij,jk,kl,im', 1/30),
                ('i,ij,jk,il,lm', 1/20),
                ('i,ij,jk,jl,jm', 1/20),
                ('i,ij,jk,kl,jm', 1/40),
                ('i,ij,jk,kl,km', 1/60),
                ('i,ij,jk,kl,lm', 1/120)],
        }
        print('Warning: methods of order >5 will still show as order 5')
        print('Computing Butcher order from Python')
        A, b = self.A, self.b
        order = 0
        for o, conds in conditions.items():
            for sSum, val in conds:
                s = np.einsum(sSum, b, *[A]*sSum.count(','), optimize="greedy").sum()
                if not np.isclose(s, val):
                    return order
            order = o
        return order


class SDCAnalysis(RKAnalysis):

    def __init__(self, M=3, quadType="RADAU-RIGHT", distr="LEGENDRE",
                 sweepList=2*["BE"], preSweep="BE", postSweep="LASTNODE",
                 lowStorage=False):
        from numpy import tril
        # Collocation coefficients
        self.quadType = quadType
        self.postSweep = postSweep
        nodes, weights, Q = genCollocation(M, distr, quadType)
        if lowStorage:
            self.QTri = tril(Q, -1)
        else:
            self.QTri = np.zeros(M)
        self.nodes, self.weights, self.Q = nodes, weights, Q
        # SDC coefficients
        if preSweep == "COPY": preSweep = "PIC"
        QDeltaInit, dtau = genQDelta(nodes, preSweep, Q)
        QDeltaList = [genQDelta(nodes, sType, Q)[0] for sType in sweepList]
        self.QDeltaInit, self.dtau, self.QDeltaList = QDeltaInit, dtau, QDeltaList
        self.lowStorage=lowStorage

        # Build Butcher table
        A, b, c = self.buildTable(QDeltaList, QDeltaInit)
        # Call parent constructor
        super().__init__(A, b, c)

    def buildTable(self, QDeltaList, QDeltaInit):
        Q = self.Q
        QTri = self.QTri

        #print("This is Q:")
        #print(self.Q)
        #print("This is QDelta")
        #print(QDeltaList)
        nodes = self.nodes 
        weights = self.weights
        postSweep = self.postSweep
        quadType = self.quadType
        dtau = self.dtau
        zeros = np.zeros_like(Q)
        nSweep = len(QDeltaList)
        # -- build A
        A = np.hstack([QDeltaInit] + nSweep*[zeros])
        for k, QDelta in enumerate(QDeltaList):
            A = np.vstack(
                [A,
                 np.hstack(k*[zeros]
                           + [Q - QDelta - QTri, QDelta + QTri]
                           + (nSweep-k-1)*[zeros])
                ])
        # -- build b and c
        zeros = np.zeros_like(nodes)
        #self.postSweep = postSweep
        if postSweep == "QUADRATURE":
            b = np.hstack(nSweep*[zeros]+[weights])
        elif postSweep == "LASTNODE":
            if quadType in ["GAUSS", "RADAU-LEFT"]:
                raise ValueError("cannot use LASTNODE with GAUSS or RADAU-LEFT")
            b = A[-1]
        else:
            b = np.hstack(nSweep*[zeros]+[weights])
            #raise NotImplementedError(f"postSweep={postSweep}")
            pass
        c = np.hstack((nSweep+1)*[nodes])
        # -- add dtau term if needed
        if np.any(dtau != 0):
            newA = np.zeros([s+1 for s in A.shape])
            newA[1:, 1:] = A
            newA[1:self.M+1, 0] = dtau
            A = newA
            b = np.hstack([0, b])
            c = np.hstack([0, c])
        return A, b, c


    @property
    def nSweep(self):
        return len(self.QDeltaList)

    @property
    def M(self):
        return len(self.nodes)


    def numericalSolution(self, z, u0=1):
        """
        Compute the numerical solution  of one step for 
        the Dahlquist test equation using SDC matrices

        Parameters
        ----------
        z : scalar or vector or matrix or ...
            Complex values where to evaluate the stability function.
        u0 : scalar, optional
            Value for the initial solution. The default is 1.

        Returns
        -------
        u : scalar or vector or matrix or ...
            Numerical solution evaluated on the complex values.
        """
        # Pre-processing for z values
        z = np.asarray(z)
        shape = z.shape
        z = z.ravel()[:, None, None]    # shape (nValues, 1, 1)

        print(f"Computing SDC numerical solution for {z.size} values")

        # Prepare SDC matrices to shape (1, M, M)
        Q = self.Q - self.QTri
        Q = self.Q[None, :, :]
        QDeltaInit = self.QDeltaInit[None, :, :]
        temp = [QDelta + self.QTri for QDelta in self.QDeltaList]
        QDeltaList = [QDelta[None, :, :] for QDelta in temp]

        # Utilities
        uInit = u0*np.ones(self.M)[None, :]     # shape (1, M)
        I = np.eye(self.M)[None, :, :]          # shape (1, M, M)

        # Pre-sweep : u^{0} = (I-zQDeltaI)^{-1} @ (1+z*dtau)*uInit
        rhs = (1 + z[:, 0]*self.dtau)*uInit     # shape (nValues, M)
        u = matVecInv(I - z*QDeltaInit, rhs)    # shape (nValues, M)

        # Sweeps u^{k+1} = (I-zQDelta)^{-1} @ z(Q-QDelta) @ u^{k+1}
        #                  + (I-zQDelta)^{-1} @ uInit
        for QDelta in QDeltaList:
            L = I - z*QDelta            # shape (nValues, M, M)
            u = matVecInv(L, matVecMul(z*(Q - QDelta), u))
            u += matVecInv(L, uInit)    # shape (nValues, M)

        # Post-sweep
        if self.postSweep == "QUADRATURE":
            u = u0 + z*self.weights[None, None, :] @ u[:, :, None]
        elif self.postSweep == "LASTNODE":
            h = np.zeros(self.M)
            h[-1] = 1
            u = h[None, None, :] @ u[:, :, None]
        elif self.postSweep == "EXTRAPOLATE":
            L = LagrangeApproximation(self.nodes)
            h = L.getInterpolationMatrix(1*np.ones(np.shape(z.flatten())))
            u = np.sum(h*u, axis=1)    
            #h = np.zeros(self.M)
            #h[-1] = 1
            #u = h[None, None, :] @ u[:, :, None]              

        # Reshape to original size (scalar or vector)
        u.shape = shape
        return u
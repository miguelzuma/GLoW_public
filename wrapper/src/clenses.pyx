#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np

import cython
cimport cython
from cython.parallel import prange

cimport clenses

## ----------  Precision ------------- ##
handle_GSL_errors()
cdef update_pprec(Prec_General pprec_general):
    global pprec
    pprec = pprec_general

try:
    import psutil
    max_num_threads = psutil.cpu_count(logical=False)
except ModuleNotFoundError:
    import multiprocessing
    max_num_threads = int(multiprocessing.cpu_count()/2)
## ----------------------------------- ##


## =======     LENSES STUFF
## =============================================================================

cdef pNamedLens* convert_pphys_to_pLens_SIS(Psi):
    cdef double psi0 = Psi.p_phys['psi0']

    return clenses.create_pLens_SIS(psi0)

cdef pNamedLens* convert_pphys_to_pLens_CIS(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double rc   = Psi.p_phys['rc']

    return clenses.create_pLens_CIS(psi0, rc)

cdef pNamedLens* convert_pphys_to_pLens_PointLens(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double xc   = Psi.p_prec['xc']

    return clenses.create_pLens_PointLens(psi0, xc)

cdef pNamedLens* convert_pphys_to_pLens_Ball(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double b    = Psi.p_phys['b']

    return clenses.create_pLens_Ball(psi0, b)

cdef pNamedLens* convert_pphys_to_pLens_NFW(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double xs   = Psi.p_phys['xs']

    return clenses.create_pLens_NFW(psi0, xs)
    
cdef pNamedLens* convert_pphys_to_pLens_tSIS(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double xb   = Psi.p_phys['xb']

    return clenses.create_pLens_tSIS(psi0, xb)

cdef pNamedLens* convert_pphys_to_pLens_offcenterSIS(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double xc1  = Psi.p_phys['xc1']
    cdef double xc2  = Psi.p_phys['xc2']

    return clenses.create_pLens_offcenterSIS(psi0, xc1, xc2)

cdef pNamedLens* convert_pphys_to_pLens_offcenterCIS(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double rc   = Psi.p_phys['rc']
    cdef double xc1  = Psi.p_phys['xc1']
    cdef double xc2  = Psi.p_phys['xc2']

    return clenses.create_pLens_offcenterCIS(psi0, rc, xc1, xc2)

cdef pNamedLens* convert_pphys_to_pLens_offcenterPointLens(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double xc   = Psi.p_prec['xc']
    cdef double xc1  = Psi.p_phys['xc1']
    cdef double xc2  = Psi.p_phys['xc2']

    return clenses.create_pLens_offcenterPointLens(psi0, xc, xc1, xc2)

cdef pNamedLens* convert_pphys_to_pLens_offcenterBall(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double b    = Psi.p_phys['b']
    cdef double xc1  = Psi.p_phys['xc1']
    cdef double xc2  = Psi.p_phys['xc2']

    return clenses.create_pLens_offcenterBall(psi0, b, xc1, xc2)

cdef pNamedLens* convert_pphys_to_pLens_offcenterNFW(Psi):
    cdef double psi0 = Psi.p_phys['psi0']
    cdef double xs   = Psi.p_phys['xs']
    cdef double xc1  = Psi.p_phys['xc1']
    cdef double xc2  = Psi.p_phys['xc2']

    return clenses.create_pLens_offcenterNFW(psi0, xs, xc1, xc2)

cdef pNamedLens* convert_pphys_to_pLens_CombinedLens(Psi):
    cdef pNamedLens *combined_pNLens
    cdef pNamedLens *new_pNLens

    sublenses = Psi.p_phys['lenses']
    n_sublenses = len(sublenses)

    combined_pNLens = clenses.create_pLens_CombinedLens(n_sublenses)

    for l in sublenses:
        new_pNLens = convert_pphys_to_pLens(l)
        clenses.add_lens_CombinedLens(new_pNLens, combined_pNLens)

    return combined_pNLens

cdef pNamedLens* convert_pphys_to_pLens_Grid1d(Psi):
    cdef int n_grid     = Psi.p_phys['n_grid']
    cdef bytes py_fname = Psi.p_phys['root'].encode()

    return clenses.create_pLens_Grid1d(py_fname, n_grid)

cdef pNamedLens* convert_pphys_to_pLens_SIE(Psi):
    cdef double psi0  = Psi.p_phys['psi0']
    cdef double q     = Psi.p_phys['q']
    cdef double alpha = Psi.p_phys['alpha']
    cdef double xc1   = Psi.p_phys['xc1']
    cdef double xc2   = Psi.p_phys['xc2']

    return clenses.create_pLens_SIE(psi0, q, alpha, xc1, xc2)

cdef pNamedLens* convert_pphys_to_pLens_Ext(Psi):
    cdef double kappa  = Psi.p_phys['kappa']
    cdef double gamma1 = Psi.p_phys['gamma1']
    cdef double gamma2 = Psi.p_phys['gamma2']

    return clenses.create_pLens_Ext(kappa, gamma1, gamma2)

## =============================================================================

implemented_lenses = ['SIS',
                      'CIS',
                      'point lens',
                      'ball',
                      'NFW',
                      'tSIS',
                      'off-center SIS',
                      'off-center CIS',
                      'off-center point lens',
                      'off-center ball',
                      'off-center NFW',
                      'combined lens',
                      'grid 1d',
                      'SIE',
                      'ext']

def check_implemented_lens(Psi):
    if Psi.p_phys['name'] in implemented_lenses:
        return True
    else:
        return False

cdef pNamedLens* convert_pphys_to_pLens(Psi):
    name = Psi.p_phys['name']

    if name == 'SIS':
        return convert_pphys_to_pLens_SIS(Psi)
    elif name == 'CIS':
        return convert_pphys_to_pLens_CIS(Psi)
    elif name == 'point lens':
        return convert_pphys_to_pLens_PointLens(Psi)
    elif name == 'ball':
        return convert_pphys_to_pLens_Ball(Psi)
    elif name == 'NFW':
        return convert_pphys_to_pLens_NFW(Psi)
    elif name == 'tSIS':
        return convert_pphys_to_pLens_tSIS(Psi)
    elif name == 'off-center SIS':
        return convert_pphys_to_pLens_offcenterSIS(Psi)
    elif name == 'off-center CIS':
        return convert_pphys_to_pLens_offcenterCIS(Psi)
    elif name == 'off-center point lens':
        return convert_pphys_to_pLens_offcenterPointLens(Psi)
    elif name == 'off-center ball':
        return convert_pphys_to_pLens_offcenterBall(Psi)
    elif name == 'off-center NFW':
        return convert_pphys_to_pLens_offcenterNFW(Psi)
    elif name == 'combined lens':
        return convert_pphys_to_pLens_CombinedLens(Psi)
    elif name == 'grid 1d':
        return convert_pphys_to_pLens_Grid1d(Psi)
    elif name == 'SIE':
        return convert_pphys_to_pLens_SIE(Psi)
    elif name == 'ext':
        return convert_pphys_to_pLens_Ext(Psi)
    else:
        message = "WRAPPER ERROR: Unknown lens '%s'" % name
        raise ValueError(message)

## =============================================================================

cdef class LensWrapper():
    cdef pNamedLens *pNLens
    cdef Lens Psi_C
    cdef readonly bint isAxisym
    cdef readonly bint hasDeriv1
    cdef readonly bint hasDeriv2
    cdef readonly dict p_phys
    cdef readonly dict p_prec
    cdef readonly object asymp_index
    cdef readonly object asymp_amplitude

    def __cinit__(self, Psi):
        self.pNLens = convert_pphys_to_pLens(Psi)
        self.Psi_C = init_lens(self.pNLens)
        self.p_phys = Psi.p_phys
        self.p_prec = Psi.p_prec

        self.isAxisym  = False
        self.hasDeriv1 = True
        self.hasDeriv2 = True
        self.asymp_index = Psi.asymp_index
        self.asymp_amplitude = Psi.asymp_amplitude

    cdef get_xs(self, x1, x2):
        x1s = x1
        x2s = x2
        use_vec = False
        shape_x1 = None
        shape_x2 = None

        if np.isscalar(x1s) is False:
            use_vec = True
            shape_x1 = x1s.shape
            if np.isscalar(x2s):
                x2s = np.full_like(x1s, x2s)
                shape_x2 = shape_x1

        if np.isscalar(x2s) is False:
            use_vec = True
            shape_x2 = x2s.shape
            if np.isscalar(x1s):
                x1s = np.full_like(x2s, x1s)
                shape_x1 = shape_x2

        if shape_x1 == shape_x2:
            shape = shape_x1
            if shape is not None:
                x1s = x1s.flatten()
                x2s = x2s.flatten()
        else:
            shape = -1
            raise ValueError(f"operands could not be broadcast together with shapes {x1.shape} {x2.shape}")

        return use_vec, shape, x1s, x2s

    def psi(self, x1, x2, parallel=True):
        use_vec, shape, x1s, x2s = self.get_xs(x1, x2)

        cdef int i
        cdef double[:] cx1 = np.ascontiguousarray(x1s, dtype=np.double)
        cdef double[:] cx2 = np.ascontiguousarray(x2s, dtype=np.double)
        cdef int n_xs = cx1.shape[0]
        cdef int nthreads = max_num_threads if parallel else 1

        pypsi = np.zeros(n_xs, dtype=np.double)
        cdef double[:] psi = pypsi

        for i in prange(n_xs, nogil=True, num_threads=nthreads, schedule='static'):
            psi[i] = call_psi(cx1[i], cx2[i], &self.Psi_C)

        if not use_vec:
            pypsi = pypsi.item()
        else:
            pypsi = pypsi.reshape(shape)

        return pypsi

    def psi_1stDerivs(self, x1, x2, parallel=True):
        use_vec, shape, x1s, x2s = self.get_xs(x1, x2)

        cdef int i
        cdef double[:] cx1 = np.ascontiguousarray(x1s, dtype=np.double)
        cdef double[:] cx2 = np.ascontiguousarray(x2s, dtype=np.double)
        cdef int n_xs = cx1.shape[0]
        cdef int nthreads = max_num_threads if parallel else 1

        pypsi_derivs = np.zeros((n_xs, <int>indices_derivs.N_derivs), dtype=np.double)
        cdef double[:, :] psi_derivs = pypsi_derivs

        for i in prange(n_xs, nogil=True, num_threads=nthreads, schedule='static'):
            call_psi_1stDerivs(&psi_derivs[i, 0], cx1[i], cx2[i], &self.Psi_C)

        pypsi = pypsi_derivs[:, <int>indices_derivs.i_0]
        pyd1  = pypsi_derivs[:, <int>indices_derivs.i_dx1]
        pyd2  = pypsi_derivs[:, <int>indices_derivs.i_dx2]

        if not use_vec:
            pypsi = pypsi.item()
            pyd1  = pyd1.item()
            pyd2  = pyd2.item()
        else:
            pypsi = pypsi.reshape(shape)
            pyd1  = pyd1.reshape(shape)
            pyd2  = pyd2.reshape(shape)

        return pypsi, pyd1, pyd2

    def psi_2ndDerivs(self, x1, x2, parallel=True):
        use_vec, shape, x1s, x2s = self.get_xs(x1, x2)

        cdef int i
        cdef double[:] cx1 = np.ascontiguousarray(x1s, dtype=np.double)
        cdef double[:] cx2 = np.ascontiguousarray(x2s, dtype=np.double)
        cdef int n_xs = cx1.shape[0]
        cdef int nthreads = max_num_threads if parallel else 1

        pypsi_derivs = np.zeros((n_xs, <int>indices_derivs.N_derivs), dtype=np.double)
        cdef double[:, :] psi_derivs = pypsi_derivs

        for i in prange(n_xs, nogil=True, num_threads=nthreads, schedule='static'):
            call_psi_2ndDerivs(&psi_derivs[i, 0], cx1[i], cx2[i], &self.Psi_C)

        pypsi  = pypsi_derivs[:, <int>indices_derivs.i_0]
        pyd1   = pypsi_derivs[:, <int>indices_derivs.i_dx1]
        pyd2   = pypsi_derivs[:, <int>indices_derivs.i_dx2]
        pyd11  = pypsi_derivs[:, <int>indices_derivs.i_dx1dx1]
        pyd12  = pypsi_derivs[:, <int>indices_derivs.i_dx1dx2]
        pyd22  = pypsi_derivs[:, <int>indices_derivs.i_dx2dx2]

        if not use_vec:
            pypsi = pypsi.item()
            pyd1  = pyd1.item()
            pyd2  = pyd2.item()
            pyd11 = pyd11.item()
            pyd12 = pyd12.item()
            pyd22 = pyd22.item()
        else:
            pypsi = pypsi.reshape(shape)
            pyd1  = pyd1.reshape(shape)
            pyd2  = pyd2.reshape(shape)
            pyd11 = pyd11.reshape(shape)
            pyd12 = pyd12.reshape(shape)
            pyd22 = pyd22.reshape(shape)

        return pypsi, pyd1, pyd2, pyd11, pyd12, pyd22

    def phi_Fermat(self, x1, x2, y, parallel=True):
        use_vec, shape, x1s, x2s = self.get_xs(x1, x2)

        cdef int i
        cdef double cy = y
        cdef double[:] cx1 = np.ascontiguousarray(x1s, dtype=np.double)
        cdef double[:] cx2 = np.ascontiguousarray(x2s, dtype=np.double)
        cdef int n_xs = cx1.shape[0]
        cdef int nthreads = max_num_threads if parallel else 1

        pyphi = np.zeros(n_xs, dtype=np.double)
        cdef double[:] phi = pyphi

        for i in prange(n_xs, nogil=True, num_threads=nthreads, schedule='static'):
            phi[i] = phiFermat(cy, cx1[i], cx2[i], &self.Psi_C)

        if not use_vec:
            pyphi = pyphi.item()
        else:
            pyphi = pyphi.reshape(shape)

        return pyphi

    def phi_Fermat_1stDerivs(self, x1, x2, y, parallel):
        use_vec, shape, x1s, x2s = self.get_xs(x1, x2)

        cdef int i
        cdef double cy = y
        cdef double[:] cx1 = np.ascontiguousarray(x1s, dtype=np.double)
        cdef double[:] cx2 = np.ascontiguousarray(x2s, dtype=np.double)
        cdef int n_xs = cx1.shape[0]
        cdef int nthreads = max_num_threads if parallel else 1

        pyphi_derivs = np.zeros((n_xs, <int>indices_derivs.N_derivs), dtype=np.double)
        cdef double[:, :] phi_derivs = pyphi_derivs

        for i in prange(n_xs, nogil=True, num_threads=nthreads, schedule='static'):
            phiFermat_1stDeriv(&phi_derivs[i, 0], cy, cx1[i], cx2[i], &self.Psi_C)

        pyphi = pyphi_derivs[:, <int>indices_derivs.i_0]
        pyd1  = pyphi_derivs[:, <int>indices_derivs.i_dx1]
        pyd2  = pyphi_derivs[:, <int>indices_derivs.i_dx2]

        if not use_vec:
            pyphi = pyphi.item()
            pyd1  = pyd1.item()
            pyd2  = pyd2.item()
        else:
            pyphi = pyphi.reshape(shape)
            pyd1  = pyd1.reshape(shape)
            pyd2  = pyd2.reshape(shape)

        return pyphi, pyd1, pyd2

    def phi_Fermat_2ndDerivs(self, x1, x2, y, parallel=True):
        use_vec, shape, x1s, x2s = self.get_xs(x1, x2)

        cdef int i
        cdef double cy = y
        cdef double[:] cx1 = np.ascontiguousarray(x1s, dtype=np.double)
        cdef double[:] cx2 = np.ascontiguousarray(x2s, dtype=np.double)
        cdef int n_xs = cx1.shape[0]
        cdef int nthreads = max_num_threads if parallel else 1

        pyphi_derivs = np.zeros((n_xs, <int>indices_derivs.N_derivs), dtype=np.double)
        cdef double[:, :] phi_derivs = pyphi_derivs

        for i in prange(n_xs, nogil=True, num_threads=nthreads, schedule='static'):
            phiFermat_2ndDeriv(&phi_derivs[i, 0], cy, cx1[i], cx2[i], &self.Psi_C)

        pyphi  = pyphi_derivs[:, <int>indices_derivs.i_0]
        pyd1   = pyphi_derivs[:, <int>indices_derivs.i_dx1]
        pyd2   = pyphi_derivs[:, <int>indices_derivs.i_dx2]
        pyd11  = pyphi_derivs[:, <int>indices_derivs.i_dx1dx1]
        pyd12  = pyphi_derivs[:, <int>indices_derivs.i_dx1dx2]
        pyd22  = pyphi_derivs[:, <int>indices_derivs.i_dx2dx2]

        if not use_vec:
            pyphi = pyphi.item()
            pyd1  = pyd1.item()
            pyd2  = pyd2.item()
            pyd11 = pyd11.item()
            pyd12 = pyd12.item()
            pyd22 = pyd22.item()
        else:
            pyphi = pyphi.reshape(shape)
            pyd1  = pyd1.reshape(shape)
            pyd2  = pyd2.reshape(shape)
            pyd11 = pyd11.reshape(shape)
            pyd12 = pyd12.reshape(shape)
            pyd22 = pyd22.reshape(shape)

        return pyphi, pyd1, pyd2, pyd11, pyd12, pyd22

    ## -----------------------------------------------------------------------

    def dpsi_vec(self, x1, x2, parallel=True):
        psi, d1, d2 = self.psi_1stDerivs(x1, x2, parallel)
        return d1, d2

    def ddpsi_vec(self, x1, x2, parallel=True):
        psi, d1, d2, d11, d12, d22 = self.psi_2ndDerivs(x1, x2, parallel)
        return d11, d12, d22

    def dpsi_dx1(self, x1, x2, parallel=True):
        psi, d1, d2 = self.psi_1stDerivs(x1, x2, parallel)
        return d1

    def dpsi_dx2(self, x1, x2, parallel=True):
        psi, d1, d2 = self.psi_1stDerivs(x1, x2, parallel)
        return d2

    def ddpsi_ddx1(self, x1, x2, parallel=True):
        psi, d1, d2, d11, d12, d22 = self.psi_2ndDerivs(x1, x2, parallel)
        return d11

    def ddpsi_ddx2(self, x1, x2, parallel=True):
        psi, d1, d2, d11, d12, d22 = self.psi_2ndDerivs(x1, x2, parallel)
        return d22

    def ddpsi_dx1dx2(self, x1, x2, parallel=True):
        psi, d1, d2, d11, d12, d22 = self.psi_2ndDerivs(x1, x2, parallel)
        return d12

    def dphi_Fermat_vec(self, x1, x2, y, parallel=True):
        phi, d1, d2 = self.phi_Fermat_1stDerivs(x1, x2, y, parallel)
        return d1, d2

    def ddphi_Fermat_vec(self, x1, x2, parallel=True):
        phi, d1, d2, d11, d12, d22 = self.phi_Fermat_2ndDerivs(x1, x2, 0., parallel)
        return d11, d12, d22

    def shear(self, x1, x2, parallel=True):
        d11, d12, d22 = self.ddpsi_vec(x1, x2, parallel)

        d = {}
        d['gamma1']  = 0.5*(d11 - d22)
        d['gamma2']  = d12
        d['kappa']   = 0.5*(d11 + d22)
        d['gamma']   = np.sqrt(d['gamma1']**2 + d['gamma2']**2)
        d['lambda1'] = 1 - d['kappa'] - d['gamma']
        d['lambda2'] = 1 - d['kappa'] + d['gamma']
        d['detA']    = d['lambda1']*d['lambda2']
        d['trA']     = d['lambda1'] + d['lambda2']
        d['mag']     = 1/np.abs(d['detA'])

        return d

    def __dealloc__(self):
        free_pLens(self.pNLens)

## =============================================================================
## =============================================================================

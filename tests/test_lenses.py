import numpy as np
from glow import lenses

from glow.wrapper import check_implemented_lens, LensWrapper

# -------------------------------------------------------

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def benchmark_lens():
    # benchmark lens for 2d SL
    xs = [[0.3, 0], [-0.6, 0.3], [0.3, -0.3], [0, 0]]
    psi0 = 1./len(xs)
    rc = 0.05
    Psis = [lenses.Psi_offcenterCIS({'psi0':psi0, 'rc':rc, 'xc1':x[0], 'xc2':x[1]}) for x in xs]
    Psi = lenses.CombinedLens({'lenses':Psis})
    return Psi

class TestLens():
    def __init__(self, Psi, grid=(-2, 2, 10), seed=10435):
        xmin, xmax, Nx = grid
        self.x1, self.x2 = self.create_grid(xmin, xmax, Nx, seed)

        self.Psi = Psi

        # if the lens is not implemented in python, should return None
        self.pylens_test = True
        if Psi.psi(1, 1) == None:
            self.pylens_test = False

        # if the lens is implemented in C, test it as well
        self.clens_test = check_implemented_lens(Psi)
        if self.clens_test:
            self.Psi_C = LensWrapper(Psi)

        # check python version against C version
        self.cross_test = self.pylens_test and self.clens_test

    def create_grid(self, xmin, xmax, Nx, seed):
        np.random.seed(seed)
        x1 = xmin + (xmax-xmin)*np.random.random(Nx)
        x2 = xmin + (xmax-xmin)*np.random.random(Nx)
        return x1, x2

    def df_vec_num(self, x1, x2, eval_f, dx=1e-4):
        x1_plus_dx  = x1 + dx
        x1_minus_dx = x1 - dx
        x2_plus_dx  = x2 + dx
        x2_minus_dx = x2 - dx

        plus  = eval_f(x1_plus_dx,  x2)
        minus = eval_f(x1_minus_dx, x2)
        d1 = 0.5*(plus - minus)/dx

        plus  = eval_f(x1,  x2_plus_dx)
        minus = eval_f(x1, x2_minus_dx)
        d2 = 0.5*(plus - minus)/dx

        return d1, d2

    def ddf_vec_num(self, x1, x2, eval_df, dx=1e-4):
        x1_plus_dx  = x1 + dx
        x1_minus_dx = x1 - dx
        x2_plus_dx  = x2 + dx
        x2_minus_dx = x2 - dx

        plus  = eval_df(x1_plus_dx,  x2)
        minus = eval_df(x1_minus_dx, x2)
        d11 = 0.5*(plus[0] - minus[0])/dx
        #d21 = 0.5*(plus[1] - minus[1])/dx

        plus  = eval_df(x1,  x2_plus_dx)
        minus = eval_df(x1, x2_minus_dx)
        d12 = 0.5*(plus[0] - minus[0])/dx
        d22 = 0.5*(plus[1] - minus[1])/dx

        return d11, d12, d22

    def test_df(self, x1, x2, eval_f, eval_df, dx=1e-4, epsabs=1e-5, epsrel=1e-5):
        d_num   = self.df_vec_num(x1, x2, eval_f, dx)
        d_exact = eval_df(x1, x2)

        keys  = ['d1', 'd2']

        output = {}
        for da, db, key in zip(d_exact, d_num, keys):
            output[key] = self.allclose(da, db, epsabs, epsrel)

        return output

    def test_ddf(self, x1, x2, eval_df, eval_ddf, dx=1e-4, epsabs=1e-5, epsrel=1e-5):
        d_num   = self.ddf_vec_num(x1, x2, eval_df, dx)
        d_exact = eval_ddf(x1, x2)

        keys  = ['d11', 'd12', 'd22']

        output = {}
        for da, db, key in zip(d_exact, d_num, keys):
            output[key] = self.allclose(da, db, epsabs, epsrel)

        return output

    def test_dpsi(self, x1, x2, Psi, dx=1e-4, epsabs=1e-5, epsrel=1e-5):
        return self.test_df(x1, x2, Psi.psi, Psi.dpsi_vec, dx, epsabs, epsrel)

    def test_ddpsi(self, x1, x2, Psi, dx=1e-4, epsabs=1e-5, epsrel=1e-5):
        return self.test_ddf(x1, x2, Psi.dpsi_vec, Psi.ddpsi_vec, dx, epsabs, epsrel)

    def allclose(self, arr1, arr2, epsabs, epsrel):
        diff = np.abs(arr1 - arr2)
        i_max = np.argmax(diff)
        target = epsrel*np.abs(arr1[i_max]) + epsabs
        x = (self.x1[i_max], self.x2[i_max])
        output = (diff[i_max]<target, x, target, diff[i_max])

        return output


    ## ------------------------------------------------------------------------

    def test_pylens(self, dx=1e-4, epsabs=1e-5, epsrel=1e-5):
        full_out  = self.test_dpsi(self.x1, self.x2, self.Psi)
        full_out.update(self.test_ddpsi(self.x1, self.x2, self.Psi))

        failed_tests = [key for key, item in full_out.items() if item[0] == False]

        return failed_tests, full_out

    def test_clens(self, dx=1e-4, epsabs=1e-5, epsrel=1e-5):
        full_out  = self.test_dpsi(self.x1, self.x2, self.Psi_C)
        full_out.update(self.test_ddpsi(self.x1, self.x2, self.Psi_C))

        failed_tests = [key for key, item in full_out.items() if item[0] == False]

        return failed_tests, full_out

    def test_clens_internal(self, epsabs=1e-8, epsrel=1e-5):
        psi_a                            = self.Psi_C.psi(self.x1, self.x2)
        psi_b, d1_b, d2_b                = self.Psi_C.psi_1stDerivs(self.x1, self.x2)
        psi_c, d1_c, d2_c, d11, d12, d22 = self.Psi_C.psi_2ndDerivs(self.x1, self.x2)

        full_out = {}
        full_out['psi_ab'] = self.allclose(psi_a, psi_b, epsabs, epsrel)
        full_out['psi_ac'] = self.allclose(psi_a, psi_c, epsabs, epsrel)
        full_out['d1_bc']  = self.allclose(d1_b,  d1_c, epsabs, epsrel)
        full_out['d2_bc']  = self.allclose(d2_b,  d2_c, epsabs, epsrel)

        failed_tests = [key for key, item in full_out.items() if item[0] == False]

        return failed_tests, full_out

    def test_cross(self, epsabs=1e-8, epsrel=1e-5):
        psi = self.Psi.psi(self.x1, self.x2)
        d1, d2 = self.Psi.dpsi_vec(self.x1, self.x2)
        d11, d12, d22 = self.Psi.ddpsi_vec(self.x1, self.x2)
        psi_c, d1_c, d2_c, d11_c, d12_c, d22_c = self.Psi_C.psi_2ndDerivs(self.x1, self.x2)

        full_out = {}
        full_out['psi'] = self.allclose(psi, psi_c, epsabs, epsrel)
        full_out['d1']  = self.allclose(d1,  d1_c,  epsabs, epsrel)
        full_out['d2']  = self.allclose(d2,  d2_c,  epsabs, epsrel)
        full_out['d11'] = self.allclose(d11, d11_c, epsabs, epsrel)
        full_out['d12'] = self.allclose(d12, d12_c, epsabs, epsrel)
        full_out['d22'] = self.allclose(d22, d22_c, epsabs, epsrel)

        failed_tests = [key for key, item in full_out.items() if item[0] == False]

        return failed_tests, full_out

    def full_test(self):
        print(" - Testing lens '%s':" % self.Psi.p_phys['name'])

        def _print_results(failed_tests, full_out):
            if failed_tests == []:
                print(bcolors.OKGREEN + "ok" + bcolors.ENDC)
            else:
                print(bcolors.FAIL + "failed" + bcolors.ENDC)
                for key in failed_tests:
                    flag, x, target, diff = full_out[key]
                    print("        *** %s at x=(%e, %e), max diff=%.3e, got %.3e" \
                            % (key, x[0], x[1], target, diff))

        print("   * Python implementation:")
        if self.pylens_test:
            failed_tests, full_out = self.test_pylens()
            print("     ** Numerical derivatives: ", end='')
            _print_results(failed_tests, full_out)
        else:
            print("     ** Not implemented: skipping")

        print("   * C implementation:")
        if self.clens_test:
            failed_tests, full_out = self.test_clens()
            print("     ** Numerical derivatives: ", end='')
            _print_results(failed_tests, full_out)

            failed_tests, full_out = self.test_clens_internal()
            print("     ** Internal consistency:  ", end='')
            _print_results(failed_tests, full_out)
        else:
            print("     ** Not implemented: skipping")

        if self.cross_test:
            print("   * Cross check:")
            failed_tests, full_out = self.test_cross()
            print("     ** Python/C: ", end='')
            _print_results(failed_tests, full_out)

        print('')

# -------------------------------------------------------

if __name__ == '__main__':
    grid = (-2, 2, 100)
    seed = 10435

    p_phys = {'xc1':0.1, 'xc2':0.2}
    lenses = [lenses.Psi_SIS(),
              lenses.Psi_CIS(),
              lenses.Psi_PointLens(),
              lenses.Psi_Ball(),
              lenses.Psi_NFW(),
              lenses.Psi_tSIS(),
              lenses.Psi_eSIS(),
              lenses.Psi_eSIS({'q':1.2, 'alpha':0.3, 'xc1':0.1, 'xc2':0.2}),
              lenses.Psi_Ext({'kappa':0.12, 'gamma1':0.3, 'gamma2':0.2}),
              lenses.Psi_offcenterSIS(p_phys),
              lenses.Psi_offcenterCIS(p_phys),
              lenses.Psi_offcenterPointLens(p_phys),
              lenses.Psi_offcenterBall(p_phys),
              lenses.Psi_offcenterNFW(p_phys),
              benchmark_lens()]

    for lens in lenses:
        TestLens(lens, grid, seed).full_test()

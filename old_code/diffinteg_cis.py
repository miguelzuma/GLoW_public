import numpy as np
import scipy as sp
from scipy import integrate
from scipy.special import gamma, hyp2f1, betainc
from numba import *
from numba.experimental import jitclass
from numba import types
import os

import fourier as fourier


#from diffinteg_gnfw import Node, Contour, ImgTrack, spec_DiffractionIntegral, Node_type, Contour_type, ImgTrack_type
#, DiffractionIntegral_type


#negative sign in the definition

#if rc = 0 you should use SIS!

@jit(f8(f8,f8))
def vphi_cis(x,rc):
    return -(np.sqrt(rc**2+x**2)+rc*np.log((2*rc)/(rc+np.sqrt(rc**2+x**2))))

@jit(f8(f8,f8))
def dvphi_cis(x,rc):
    '''regularized'''
    return -x/(rc + np.sqrt(rc**2 + x**2))

@jit(f8(f8,f8))
def d2vphi_cis(x,rc):
    return -(rc*(rc**2 + x**2 + rc*np.sqrt(rc**2 + x**2)))/((rc**2 + x**2)*(rc + np.sqrt(rc**2 + x**2))**2)

@jit(f8(f8,f8))
def d3vphi_cis(x,rc):
    #removed minus sign
    return  ((rc*x*(3*rc**3 + 3*rc*x**2 + 3*rc**2*np.sqrt(rc**2 + x**2) +  2*x**2*np.sqrt(rc**2 + x**2)))/((rc**2 + x**2)**2*(rc + np.sqrt(rc**2 + x**2))**3))

@jit(f8(f8,f8))
def d4vphi_cis(x,rc):
    '''4th derivative of -\psi'''
    return  -(3*rc*(2 - (rc*(2*rc**4 + 5*rc**2*x**2 + 4*x**4))/(np.sqrt(rc**2 + x**2))**5))/x**4

#TODO: ABSOLUTELY, CHANGE THE ORDER SO Y GOES FIRST!
def cis_F_wo_reg(rc, y,  f_min=0.01, f_max=100, window_transition=1,  saddle_filter_width=3,reset_fermat_pot = True, dict_output=False):
    '''master function to do the full calculation
       dict_output-> returns dictionary of output, otherwise w, F_wo_reg, lens_data,
       #TODO: include external region integration by complex method
       ODO: ABSOLUTELY, CHANGE THE ORDER SO Y GOES FIRST!
    '''

    output = {}

    lens = cis_lens(y,rc)

    #radius of the contours, assuming a conservative tau_min
    tau_rang = fourier.tau_range(f_min,f_max,window_transition=0)
    r_out = 1.1*(1.+np.sqrt(1+2.*tau_rang[1]))#solve for tau_max

    #find the images and the minimum time delay (not computing contours)
    if reset_fermat_pot:
        lens_data0, contours = compute_contours(lens, skip_contours=True)
        mag, delay, morse_ph, im_type = fourier.compute_GO(lens_data0)
        lens.phi0 = np.amin(delay) #remove the shortest delay

    lens_data, contours = compute_contours(lens, skip_contours=False, radius=r_out)

    Fs = fourier.compute_amplification(f_min, f_max, contours, lens_data,
                               window_transition=window_transition,
                               saddle_filter_width=saddle_filter_width,full_output = dict_output)

    if not dict_output:
        w,F_wo,F_wo_reg,F_go,F_go_reg = Fs
        return w, F_wo_reg, lens_data
    else:
        #NOTE: F_wo is the brute force, which should not be used!
        output['w'], output['F_wo'], output['F_wo_reg'], output['F_go'], output['F_go_reg'],  time_domain, freq_domain, freq_domain_reg, window  = Fs

        output['lens_data'] = lens_data
        output['mag'], output['tau'], output['morse_phase'], output['image_type'] = fourier.compute_GO(lens_data)

        w, F_wo, F_go, F_go_extrem, F_go_sad = freq_domain

        w, output['F_nonsing'], output['F_go_reg'], output['F_go_extrem_reg'], output['F_go_sad_reg'] = freq_domain_reg

        output['tau_uni'], output['I_full'], output['I_sing'], output['I_extrem'], output['I_sad'] = time_domain
        output['I_reg'] = output['I_full'] - output['I_sing']
        output['contours'] = contours
        output['window'] = window
        #TODO: add projected contours/time domain


        return output





def cis_lens(y,rc):
    '''creates lens with the default parameters
        rc -> core readius
        xc1 -> impact parameter
    '''

    # TODO: incorporate missing parameters
    #external convergence & shear
    ka_0 = 0.0
    ga1_0 = 0.0
    ga2_0 = 0.0

    # lens center #TODO: promote parameters to a parameter dictionary
    #xc1 = params['y']
    xc2 = 0.0

    # fiducial source position and velocity
    y1pt = 0.9
    y2pt = 0.0

    yt1pt = 0.1
    yt2pt = 0.0 #1.0

    tmin = -6.0
    tmax = 6.0

    #lens parameters
    m = 1 #mass
    xi0 = 1.0 #scale

    lens = DiffractionIntegral(ka_0, ga1_0, ga2_0, y, xc2, y1pt, y2pt, yt1pt, yt2pt, tmin, tmax, m, xi0, rc);

    return lens

def compute_contours(lens, skip_contours=False,radius=30,nn=500,eps=0.001,compute_derivs=False):
    '''compute the contour integration for a lens
       areguments are lens center xc1 and core radius (for now)
       phi0 is the contour
    '''

    images = []
    contours = []

    rc = lens.rc
    xc1 = lens.xc1

    #two cases whether we are inside or outside the caustic #TODO: find the contours in terms of the images
    #caustic, if complex then single image, automatically taken into account
    yc2 = 1+5*rc-0.5*rc**2-0.5*np.sqrt(rc)*(rc+4)**1.5


    #first case inside caustic, 3 images
    if (xc1**2 <= yc2):

        xi1 = -1.1
        xL1, xL2 = lens.EvolveImage(xi1, 0, 0, 0, 40)
        tau_L = lens.tau(xL1, xL2)

        #add xc1 center of the lens
        x_crit_rad = xc1 + np.sqrt(rc - 0.5*rc**2 - 0.5*rc*np.sqrt(rc*(rc+4)))

        xi1 = (1.+rc*x_crit_rad)/(1+rc) #between the critical curve and the usual SIS location
        xS1, xS2 = lens.EvolveImage(xi1, 0, 0, 0, 40)
        tau_S = lens.tau(xS1, xS2)

        xi1 =(xc1+rc*x_crit_rad)/(1.+rc) #between the center and the critical curve, weighted by core readius so it converges to the SIS result
        xH1, xH2 = lens.EvolveImage(xi1, 0, 0, 0, 50) #NOTE: with numba you need to call arguments in place
        tau_H = lens.tau(xH1,xH2)

        images = [(xL1,xL2),(xS1,xS2),(xH1,xH2)]
        lens_data = (lens, images)

        if skip_contours:
            return lens_data, []

        # branch starting from the vicinity of the L image and ascend in Fermat potential
        ct1 = Contour(nn, xL1, xL2, eps, lens)
        tau1, val1 = ct1.flow_contour_adaptive_steps(tau_S-1e-4, 1e-4, 1.0, 1000, 0.1, 1e-4, 1e2, 1.5)
        #branch starting from the vicinity of the center image and descend in Fermat potential
        ct2 = Contour(nn, xH1, xH2, eps, lens)
        tau2, val2 = ct2.flow_contour_adaptive_steps(tau_S+1e-4, 1e-4, -1.0, 1000, 0.1, 1e-4, 1e2, 1.5)
        # branch starting from sufficiently far away and descend in Fermat potential
        ct3 = Contour(nn, 0.0, 0.0, radius, lens, 'infinity', 0.0, 0.0)
        tau3, val3 = ct3.flow_contour_adaptive_steps(tau_S+1e-4, 1e-1*lens.tau(radius,0), -1.0, 1000, 0.1, 1e-4, 1e2, 1.5)

        contours = [(tau1,val1),(tau2,val2),(tau3,val3)]

        return lens_data, contours

    else:
        xi1 = -1.1#0.5*((xc1-y1pt) -np.sqrt((xc1-y1pt)**2+4.)), 0
        xL1, xL2 = lens.EvolveImage(xi1, 0, 0, 0, 40)
        tau_L = lens.tau(xL1, xL2)
        #print ('single image')

        images = [(xL1,xL2)]
        lens_data = (lens, images)

        if skip_contours:
            return lens_data, []

        #reduced value of eta in WL for more delicate sampling
        ct1 = Contour(nn, 0.0, 0.0, radius, lens, 'infinity', 0.0, 0.0)
        tau1, val1 = ct1.flow_contour_adaptive_steps(tau_L+1e-4, 1e-1*lens.tau(radius,0), -1.0, 1000, 0.05, 1e-4, 1e2, 1.5)
        contours = [(tau1,val1)]

        return lens_data, contours

# --------------------------------------------------------------------------- #

Node_type = deferred_type()
Contour_type = deferred_type()
ImgTrack_type = deferred_type()
DiffractionIntegral_type = deferred_type()

# --------------------------------------------------------------------------- #

spec_Node = [
              ('x1', float64),            # 1st coordinate of node position
              ('x2', float64),            # 2nd coordinate of node position
              ('lst', optional(Node_type)),        # last node
              ('nxt', optional(Node_type)),        # next node
              ('leadQ', boolean),          # whether or not the first node
              ('host_contour', optional(Contour_type)),  # indicate which contour hosts this node object
              ('tau', float64),            # Fermat potential
              ('grad_tau', float64[:]),    # gradient of the Fermat potential
              ('hessian_tau', float64[:]),    # hessian of the Fermat potential
            ]

@jitclass(spec_Node)
class Node():
    """
    Node class
    """
    def __init__(self, x1, x2, host_contour):
        """
        """
        self.x1 = x1
        self.x2 = x2
        self.leadQ = False
        self.lst = None
        self.nxt = None
        self.host_contour = host_contour

        self.tau = 0.0
        self.grad_tau = np.zeros((2), dtype=np.float64)
        self.hessian_tau = np.zeros((3), dtype=np.float64)

        self.tau = self.host_contour.host_di.tau(self.x1, self.x2)
        self.grad_tau[0], self.grad_tau[1] = self.host_contour.host_di.xtoy(self.x1, self.x2)
        self.hessian_tau[0], self.hessian_tau[1], self.hessian_tau[2] =  \
                   self.host_contour.host_di.hessian(self.x1, self.x2)

    def update_tau(self):
        """
        Update the values for the Fermat potential and its first and second order derivatives
        """
        self.tau = self.host_contour.host_di.tau(self.x1, self.x2)
        self.grad_tau[0], self.grad_tau[1] = self.host_contour.host_di.xtoy(self.x1, self.x2)
        self.hessian_tau[0], self.hessian_tau[1], self.hessian_tau[2] =  \
                   self.host_contour.host_di.hessian(self.x1, self.x2)


    def set_lst(self, value):
        self.lst = value

    def set_nxt(self, value):
        self.nxt = value

    def set_lead(self):
        self.leadQ = True

    def append_to(self, x1, x2):
        """
        Create a new node whose coordinates are (x1, x2)
          and append it after this node
        """
        new_node = Node(x1, x2, self.host_contour)
        if self.nxt is not None:
            nxt_node = self.nxt
        self.set_nxt(new_node)
        nxt_node.set_lst(new_node)
        new_node.set_nxt(nxt_node)
        new_node.set_lst(self)

    def delink_after(self):
        """
        Delete the node linked after this one
        Keep the remaining nodes in the loop in the same order as before
        """
        node_to_drop = self.nxt
        next_node = node_to_drop.nxt
        if node_to_drop.leadQ is True:
            print("delink_after(): Warning: a leading node cannot be removed!")
        else:
            self.set_nxt(next_node)
            next_node.set_lst(self)

    def flow_step(self, dtau, t1, t2):
        """
        Flow the node to change the value of the Fermat potential tau by a given amount dtau
        The direction of the flow is given by the (unnormalized) vector t = (t1, t2)
        This function uses second-order (quadratic) flow
        """
        # first order flow
        f1, f2 = self.grad_tau
        h11, h12, h22 = self.hessian_tau
        tnorm = np.sqrt(t1**2 + t2**2)
        t1_norm = t1/tnorm
        t2_norm = t2/tnorm
        lin = t1_norm*f1 + t2_norm*f2
        quad = h11*t1_norm**2 + 2.0*h12*t1_norm*t2_norm + h22*t2_norm**2
        Del = lin**2 + 2*quad*dtau

        quad_eps = 1e-30
        quad_abs = np.abs(quad)

        # second order step
        if quad_abs <= quad_eps:
            dx = dtau/lin
        elif quad > quad_eps:
            if dtau > 0:
                dx = (np.sqrt(Del) -  lin)/quad
            elif dtau < 0:
                if Del > 0:
                    dx = (np.sqrt(Del) -  lin)/quad
                else:
                    dx = dtau/lin
            else:
                dx = 0.0
        elif quad < - quad_eps:
            if dtau < 0:
                dx = (np.sqrt(Del) -  lin)/quad
            elif dtau > 0:
                if Del > 0:
                    dx = (np.sqrt(Del) -  lin)/quad
                else:
                    dx = dtau/lin
            else:
                dx = 0.0


        self.x1 += dx*t1_norm
        self.x2 += dx*t2_norm
        self.update_tau()

    def flow_step_linear(self, dtau, t1, t2):
        """
        Flow the node to change the value of the Fermat potential tau by a given amount dtau
        The direction of the flow is given by the (unnormalized) vector t = (t1, t2)
        This function uses first-order (linear) flow
        """
        #NOTE: routine can be simplified

        f1, f2 = self.grad_tau
        h11, h12, h22 = self.hessian_tau
        tnorm = np.sqrt(t1**2 + t2**2)
        t1_norm = t1/tnorm
        t2_norm = t2/tnorm
        lin = t1_norm*f1 + t2_norm*f2

        dx = dtau/lin


        self.x1 += dx*t1_norm
        self.x2 += dx*t2_norm
        self.update_tau()

    def flow_to(self, tau, epsabs=1e-10, maxiter_quad=3, maxiter_linear=10):
        """
        Flow the node along the steepest ascending/descending path
           onto the contour of constant Fermat potential tau
        """

        # 2nd order flow
        count = 0
        while True:
            dtau = tau - self.tau
            if np.abs(dtau) < epsabs:
                return
            self.flow_step(dtau, self.grad_tau[0], self.grad_tau[1])
            count += 1
            if count > maxiter_quad:
                break

        # first order flow
        count = 0
        while True:
            dtau = tau - self.tau
            if np.abs(dtau) < epsabs:
                return
            self.flow_step_linear(dtau, self.grad_tau[0], self.grad_tau[1])
            count += 1
            if count > maxiter_linear:
                break


    def get_cr(self):
        """
        Calculate the vector pointing to the local curvature center of the constant tau contour
        """
        g1, g2 = self.grad_tau
        g_norm = np.sqrt(g1**2 + g2**2)

        h11, h12, h22 = self.hessian_tau

        # unit vector pointing toward grad(tau)
        n1, n2 = g1/g_norm, g2/g_norm
        # unit vector tangent to the constant-tau contour
        t1, t2 = -n2, n1
        # curvature radius
        cr = 2.0*g_norm/np.abs(h11*t1**2 + 2.0*h12*t1*t2 + h22*t2**2)
        return cr

Node_type.define(Node.class_type.instance_type)

# --------------------------------------------------------------------------- #

spec_Contour = [
              ('node_list', optional(Node_type)),  # a linked list of nodes; point to the first node
              ('n_node', int32),                   # total number of linked nodes
              ('xc1', float64),                   # 1st coordinate of center
              ('xc2', float64),                   # 2nd coordinate of center
              ('host_di', optional(DiffractionIntegral_type)), # indicate which DiffractionIntegral class instance hosts this Contour instance
              ]

@jitclass(spec_Contour)
class Contour():
    """
    Contour class
    """
    def __init__(self, nini, xc1, xc2, eps, host_di, enclose_type='extremum', ellipticity=0.0, ellip_position_angle=0.0):
        """
        Initialize nini nodes
        All nodes lie on a small ellipse which centers at (xc1, xc2) and has a size proportional to eps
        """
        self.xc1 = xc1
        self.xc2 = xc2

        self.node_list = None
        self.n_node = 0
        self.host_di = host_di

        if enclose_type =='extremum':

            h11, h12, h22 = self.host_di.hessian(xc1, xc2)
            Del = h12**2 + (h11 - h22)**2/4
            lam1 =   (h11 + h22)/2 + np.sqrt(Del)
            lam2 =   (h11 + h22)/2 - np.sqrt(Del)

            # convention is that lam1 < lam2
            if lam1 >= lam2:
                xx = lam2
                lam2 = lam1
                lam1 = xx

            #print(lam1, lam2)

            if lam1 > 0:
                a = lam1**(-0.5)
                b = lam2**(-0.5)
            else:
                a = (-lam1)**(-0.5)
                b = (-lam2)**(-0.5)

            norm = np.sqrt(a**2 + b**2)
            a = a/norm
            b = b/norm

            delta = 0.5*np.angle((h11 - h22)/2 + 1.0j*h12) + np.pi/2
            cd = np.cos(delta)
            sd = np.sin(delta)

            theta = np.linspace(0, 2*np.pi, nini+1)[:-1]
            cth = np.cos(theta)
            sth = np.sin(theta)
            x1 = xc1 + eps*(a*cth*cd - b*sth*sd)
            x2 = xc2 + eps*(b*sth*cd + a*cth*sd)

        elif enclose_type == 'cusp':

            theta = np.linspace(0, 2*np.pi, nini+1)[:-1]
            cth = np.cos(theta)
            sth = np.sin(theta)

            phi = ellip_position_angle
            cphi = np.cos(phi)
            sphi = np.sin(phi)

            a = 1.0
            b = np.sqrt(1.0 - ellipticity**2)

            x1 = xc1 + eps*(a*cth*cphi - b*sth*sphi)
            x2 = xc2 + eps*(b*sth*cphi + a*cth*sphi)

        elif enclose_type == 'infinity':

            theta = np.linspace(0, 2*np.pi, nini+1)[:-1]
            cth = np.cos(theta)
            sth = np.sin(theta)

            x1 = xc1 + eps*cth
            x2 = xc2 + eps*sth

        for i in range(nini):
                # add the first node
                if i == 0:
                    new_node = Node(x1[i], x2[i], self)
                    self.n_node += 1
                    self.node_list = new_node
                    self.node_list.set_lead()
                    self.node_list.set_lst(self.node_list)
                    self.node_list.set_nxt(self.node_list)
                    cnd = self.node_list
                else:
                    cnd.append_to(x1[i], x2[i])
                    self.n_node += 1
                    cnd = cnd.nxt

    def increase_n_node(self, m):
        """
        Increase the number of hosted nodes by m
        """
        self.n_node += m

    def flow_contour(self, tau, nstep=10):
        """
        Flow all nodes along the contour such that the contour has a constant Fermat potential tau
        """
        cnd = self.node_list

        tau0 = cnd.tau

        #tau_list = tau0 + np.sqrt(np.linspace(0, (tau-tau0)**2, nstep+1)[1:])
        tau_list = np.linspace(tau0, tau, nstep+1)[1:]
        integ_res = np.zeros((nstep), dtype=np.float64)

        for i in range(len(tau_list)):

            self.contour_thin()

            self.contour_refine()

            tau_aim = tau_list[i]

            #print(self.n_node)

            while True:
                cnd.flow_to(tau_aim)
                cnd = cnd.nxt
                if cnd.leadQ is True:
                    break

            integ_res[i] = self.contour_integration()

        return tau_list, integ_res

    def flow_contour_adaptive_steps(self, tau, dtau0, dtau_direction, max_n_steps=10000, eta=0.1, dtau_min=1e-4, dtau_max=0.1, step_resize=1.5, compute_derivs = False):
        """
        Flow all nodes along the contour such that the contour has a constant Fermat potential tau
        Target Fermat potential value is given by tau
        Initially step size is given by dtau0; adaptive step size must be in the range [dtau_min, dtau_max]
        dtau_direction: 1.0 for increasing tau; -1.0 for decreasing tau
        """
        cnd = self.node_list
        tau0 = cnd.tau

        eps = 1e-30

        n_steps = 0

        tau_list = np.zeros((max_n_steps), dtype=np.float64)
        integ_res = np.zeros((max_n_steps), dtype=np.float64)

        tau_list[0] = tau0
        integ_res[0] = self.contour_integration()

        dtau_last = dtau0

        while n_steps < max_n_steps and dtau_direction*(tau_list[n_steps] - tau) < 0.0:

            # determine step size
            if n_steps >= 2:
                t0, t1, t2 = tau_list[n_steps-2:n_steps+1]
                v0, v1, v2 = integ_res[n_steps-2:n_steps+1]

                vel = -((t1**2*v0 - 2*t1*t2*v0 + t2**2*v0 - t0**2*v1 + 2*t0*t2*v1 - t2**2*v1 +  \
                         t0**2*v2 - t1**2*v2 - 2*t0*t2*v2 + 2*t1*t2*v2)/((t0 - t1)*(t0 - t2)*(t1 - t2)))
                acc = (-2*(-(t1*v0) + t2*v0 + t0*v1 - t2*v1 - t0*v2 + t1*v2))/((t0 - t1)*(t0 - t2)*(t1 - t2))

                dtau_est = eta*np.abs(vel)/(np.abs(acc) + eps)

                #print(dtau_est, dtau_min, dtau_max)

                dtau_1 = min(dtau_est, step_resize*dtau_last)
                dtau_last = max(dtau_1, dtau_min)
                dtau = dtau_direction*dtau_last
            else:
                dtau_last = dtau0
                dtau = dtau_direction*dtau_last

            self.contour_thin()
            self.contour_refine()
            tau_aim = tau_list[n_steps] + dtau

            #print(n_steps, tau_aim, dtau)

            cnd = self.node_list
            while True:
                cnd.flow_to(tau_aim)
                cnd = cnd.nxt
                if cnd.leadQ is True:
                    break

            n_steps = n_steps + 1
            tau_list[n_steps] = tau_aim
            integ_res[n_steps] = self.contour_integration(compute_derivs)

        # return final results
        return tau_list[1:n_steps], integ_res[1:n_steps]

    def contour_thin(self, delta_min=0.005, L_min=0.0, n_node_min=10, cr_max=1.0):
        """
        For three successive nodes N1, N2 and N3, if the distance between N1 and N3 is
           smaller than delta_min times the local curvature radius of the tau contour at N2,
           then we remove the middle node N2
        """
        # No need to do anything if the contour has very few number of nodes
        if self.n_node < n_node_min:
            return

        cnd1 = self.node_list
        cnd2 = cnd1.nxt
        cnd3 = cnd2.nxt

        while (cnd2.leadQ==False) and (cnd3.leadQ==False):

            s13 = np.sqrt((cnd1.x1 - cnd3.x1)**2 + (cnd1.x2 - cnd3.x2)**2)
            cr = min(cnd2.get_cr(), cr_max)

            if s13 < delta_min*cr or s13 < L_min:
                cnd1.delink_after()
                self.n_node -= 1
                cnd2 = cnd3
                cnd3 = cnd3.nxt
            else:
                cnd1 = cnd2
                cnd2 = cnd3
                cnd3 = cnd3.nxt

    def contour_refine(self, delta_max=0.02, L_max=10.0):
        """
        For two successive nodes N1 and N2, if the distance between N1 and N2 is larger than
           delta_max times the local curvature radius of the tau contour at N2, then we insert
           a new node at the mid-point in between N1 and N2
        """

        cnd1 = self.node_list
        cnd2 = cnd1.nxt

        while True:

            s12 = np.sqrt((cnd1.x1 - cnd2.x1)**2 + (cnd1.x2 - cnd2.x2)**2)
            cr = cnd2.get_cr()

            if s12 > delta_max*cr or s12 > L_max:
                cnd1.append_to((cnd1.x1 + cnd2.x1)/2, (cnd1.x2 + cnd2.x2)/2)
                self.n_node += 1
                cnd1 = cnd2
                cnd2 = cnd2.nxt
            else:
                cnd1 = cnd2
                cnd2 = cnd2.nxt

            if cnd1.leadQ is True:
                break

    def contour_integration(self,compute_derivs = False):
        """
        Evaluate the circumference of the contour inversely weighted by the local gradient of the Fermat potential
        compute_derivs -> compute the fisher derivatives
        """
        cnd1 = self.node_list
        cnd2 = cnd1.nxt

        if self.n_node < 10:
            return 0.0

        res = 0.0
        res_phi = 0.0

        #count = 0

        while True:

            s12 = np.sqrt((cnd1.x1 - cnd2.x1)**2 + (cnd1.x2 - cnd2.x2)**2)
            w1 = 1.0/np.sqrt(cnd1.grad_tau[0]**2 + cnd1.grad_tau[1]**2)
            w2 = 1.0/np.sqrt(cnd2.grad_tau[0]**2 + cnd2.grad_tau[1]**2)

            res += s12*(w1 + w2)/2

            if compute_derivs:
                res_phi = s12*(w1*cnd1.tau + w2*cnd2.tau)/2
                print('derivs need implementing!')
                #break

            #count += 1

            cnd1 = cnd2
            cnd2 = cnd2.nxt



            if cnd1.leadQ is True:
                break

        #print('sum over terms:', count)

        return res



Contour_type.define(Contour.class_type.instance_type)

# --------------------------------------------------------------------------- #

spec_ImgTrack = [
                  ('t_list', float64[:]),           # sequence of epochs [unit time]
                  ('x_list', float64[:, :]),        # sequence of image positions [Einstein angle of unit mass]
                  ('mu_list', float64[:]),          # sequence of signed magnification
                  ('lst', optional(ImgTrack_type)),        # last image track
                  ('nxt', optional(ImgTrack_type))         # next image track
                 ]

@jitclass(spec_ImgTrack)
class ImgTrack():
    """
    A sequence of images corresponding to linear, monotonic motion of a point source
    A single sequence of images is defined such that it does not cross any critical curve
    """
    def __init__(self):
        """
        """
        self.lst = None
        self.nxt = None
        #self.ptsrc = ptsrc

    def set_lst(self, value):
        self.lst = value

    def set_nxt(self, value):
        self.nxt = value

ImgTrack_type.define(ImgTrack.class_type.instance_type)

## --------------------------------------------------------------------------- #

spec_DiffractionIntegral = [('ka_0', float64),    # external convergence
                            ('ga1_0', float64),   # 1st component of external shear
                            ('ga2_0', float64),   # 2nd component of external shear
                            ('xc1', float64),     # 1st coordinate of lens center
                            ('xc2', float64),     # 2nd coordinate of lens center
                            ('y1pt', float64),    # 1st coordinate of fiducial source position at t = 0
                            ('y2pt', float64),    # 2nd coordinate of fiducial source position at t = 0
                            ('yt1pt', float64),    # 1st component of fiducial source velocity
                            ('yt2pt', float64),    # 2nd component of fiducial source velocity
                            ('tmin', float64),    # minimum time
                            ('tmax', float64),     # maximum time
                            ('xini_list', float64[:, :]),      # a list of image positions   for the initial time source position
                            ('xfin_list', float64[:, :]),                  # a list of  image positions for the final time source position
                            ('xini_imgtrack_exist', boolean[:]),   # flags to indicate if    each initial-time image position is the end point of an exisiting image track
                            ('xfin_imgtrack_exist', boolean[:]),   # flags to indicate if    each final-time image position is the end point of an exisiting image track
                            ('img_tracks', optional(ImgTrack_type)),   # a linked list of image tracks
                            ('m', float64),       # lens mass normalization constant
                            ('xi0', float64),     # lens scale normalization constant
                            ('rc', float64),    # core size
                            ('phi0', float64),    # lens potential offset
                           ]

@jitclass(spec_DiffractionIntegral)
class DiffractionIntegral():
    """
    Evaluating the diffraction integral
    """
    def __init__(self, ka_0=0.0, ga1_0=0.0, ga2_0=0.0, xc1=1.0, xc2=0.0,  \
                 y1pt=0.0, y2pt=0.0, yt1pt=1.0, yt2pt=0.0, tmin=-10.0, tmax=10.0, m=1.0, xi0=1.0, \
                 rc=0):
        """
        """
        # external convergence and shear
        self.ka_0 = ka_0
        self.ga1_0 = ga1_0
        self.ga2_0 = ga2_0

        # lens center
        self.xc1 = xc1
        self.xc2 = xc2

        # fiducial source (used to find ray-optics image positions)
        self.y1pt = y1pt
        self.y2pt = y2pt
        self.yt1pt = yt1pt
        self.yt2pt = yt2pt
        self.tmin, self.tmax = tmin, tmax

        # image tracks
        self.img_tracks = None

        # lens mass and scale normalizations
        self.m = m
        self.xi0 = xi0

        #offset of the time delay
        self.phi0 = 0

        #core radius
        self.rc = rc


    def getyptoft(self, t):
        """
        Fiducial image position at time t
        """
        return self.y1pt + t*self.yt1pt, self.y2pt + t*self.yt2pt

    def is_regular(self,x1,x2,eps=1e-7):
        """
        check if the lens is regular at that point
        """
        if self.rc ==0 and x1**2 + x2**2 ==0:
            return False
        else:
            return True

    def phi(self, x1, x2):
        """
        lensing potential
        lens center at (xc1, xc2)
        added offset phi0
        """
        X1 = (x1 - self.xc1)/self.xi0
        X2 = (x2 - self.xc2)/self.xi0
        return -self.m*vphi_cis(np.sqrt(X1**2 + X2**2),self.rc) +self.phi0

    def dphi(self, x1, x2):
        """
        Gradient of lensing potential gives the defletion (alp1, alp2)
        """
        X1 = (x1 - self.xc1)/self.xi0
        X2 = (x2 - self.xc2)/self.xi0
        R = np.sqrt(X1**2 + X2**2)
        dvphi = dvphi_cis(R,self.rc)
        return -self.m/self.xi0*X1/R*dvphi, -self.m/self.xi0*X2/R*dvphi

    def ddphi(self, x1, x2):
        """
        2nd-order gradient of the lensing potential is a symmetric matrix:
            (d11, d12, d22)
        """
        X1 = (x1 - self.xc1)/self.xi0
        X2 = (x2 - self.xc2)/self.xi0
        R = np.sqrt(X1**2 + X2**2)
        dvphi = dvphi_cis(R,self.rc)
        d2vphi = d2vphi_cis(R,self.rc)
        d11 = -self.m/self.xi0**2*(d2vphi*X1*X1/R**2 + dvphi*X2*X2/R**3)
        d12 = -self.m/self.xi0**2*(d2vphi/R**2 - dvphi/R**3)*X1*X2
        d22 = -self.m/self.xi0**2*(d2vphi*X2*X2/R**2 + dvphi*X1*X1/R**3)
        return d11, d12, d22

    def dddphi(self, x1, x2):
        """
        3rd-order gradient of the lensing potential has 4 independent components:
            (d111, d112, d122, d222)
        """
        X1 = (x1 - self.xc1)/self.xi0
        X2 = (x2 - self.xc2)/self.xi0
        R = np.sqrt(X1**2 + X2**2)
        dvphi = dvphi_cis(R,self.rc)
        d2vphi = d2vphi_cis(R,self.rc)
        d3vphi = d3vphi_cis(R,self.rc)
        d111 = -self.m/self.xi0**3*(d3vphi*X1**3/R**3 + 3.*d2vphi*X1*X2**2/R**4 - 3.*dvphi*X1*X2**2/R**5)
        d112 = -self.m/self.xi0**3*(d3vphi*X1**2*X2/R**3 + d2vphi*X2*(X2**2-2.*X1**2)/R**4 + dvphi*(2.*X1**2-X2**2)*X2/R**5)
        d122 = -self.m/self.xi0**3*(d3vphi*X2**2*X1/R**3 + d2vphi*X1*(X1**2-2.*X2**2)/R**4 + dvphi*(2.*X2**2-X1**2)*X1/R**5)
        d222 = -self.m/self.xi0**3*(d3vphi*X2**3/R**3 + 3.*d2vphi*X2*X1**2/R**4 - 3.*dvphi*X2*X1**2/R**5)
        return d111, d112, d122, d222

    def d1phi(self, R):
        x = dvphi_cis(R,self.rc)
        return x

    def d2phi(self, R):
        x = d2vphi_cis(R,self.rc)
        return x

    def d3phi(self, R):
        x = d3vphi_cis(R,self.rc)
        return x

    def d4phi(self, R):
        x = d4vphi_cis(R,self.rc)
        return x


    def xtoy(self, x1, x2):
        """
        Map image position (x1, x2) to the source position (y1, y2) via the lens equation
        Note that this also gives the gradient of the Fermat potential tau(x)
        """
        alp1, alp2 = self.dphi(x1, x2)
        ka = self.ka_0
        ga1 = self.ga1_0
        ga2 = self.ga2_0
        return (1.0 - ka - ga1)*x1 - ga2*x2 - alp1, (1.0 - ka + ga1)*x2 - ga2*x1 - alp2

    def hessian(self, x1, x2):
        """
        hessian is a symmetric matrix: (h11, h12, h22)
        """
        d11, d12, d22 = self.ddphi(x1, x2)
        ka = self.ka_0
        ga1 = self.ga1_0
        ga2 = self.ga2_0
        h11 = 1.0 - ka - ga1 - d11
        h12 =          - ga2 - d12
        h22 = 1.0 - ka + ga1 - d22
        return h11, h12, h22

    def hessian3(self, x1, x2):
        """
        This gives the gradient of the hessian matrix which has 4 independent components
        """
        d111, d112, d122, d222 = self.dddphi(x1, x2)
        return -d111, -d112, -d122, -d222

    def jac(self, x1, x2):
        """
        Jacobian is the determinant of the hessian matrix
        """
        h11, h12, h22 = self.hessian(x1, x2)
        return h11*h22 - h12*h12

    def invhessian(self, x1, x2):
        """
        The inverse of the hessian matrix which is also symmetric:
            (ih11, ih12, ih22)
        """
        h11, h12, h22 = self.hessian(x1, x2)
        jac = self.jac(x1, x2)
        ih11 = h22/jac
        ih12 = -h12/jac
        ih22 = h11/jac
        return ih11, ih12, ih22

    def tau(self, x1, x2):
        """
        Fermat potential
        """
        return 0.5*(   (1 - self.ka_0 - self.ga1_0)*x1**2  \
                     + (1 - self.ka_0 + self.ga1_0)*x2**2  \
                     - 2.0*self.ga2_0*x1*x2                \
                   ) - self.phi(x1, x2)

    def dxdlam_solve_ray_eqn(self, x1, x2, u1, u2):
        """
        At fixed time t, evolve the image position x = (x1, x2)
          according to dx / dlambda = J^-1 * u
        Here u = (u1, u2) is a two-component constant vector of source-position flow
        J is the two-by-two Jacobian matrix evaluated at image position x
        lambda is the flow parameter in [0, 1]
        """
        ih11, ih12, ih22 = self.invhessian(x1, x2)
        return ih11*u1 + ih12*u2, ih12*u1 + ih22*u2

    def EvolveImage(self, x1_0, x2_0, y1, y2, niter=10, eps=1.0, etol=1e-14):
        """
        Iteratively find the image position that maps to the source position y = (y1, y2)
        Use image position x_0 = (x1_0, x2_0) as an intial guess
        """
        x1, x2 = x1_0, x2_0
        # iteratively improve the guess for the solution x
        count = 0
        while True:
            y1_0, y2_0 = self.xtoy(x1, x2)
            if np.sqrt((y1 - y1_0)**2 + (y2 - y2_0)**2) < etol or count >= niter:
                break
            u1 = (y1 - y1_0)*eps
            u2 = (y2 - y2_0)*eps
            dx1, dx2 =  self.dxdlam_solve_ray_eqn(x1, x2, u1, u2)
            x1 = x1 + dx1
            x2 = x2 + dx2
            count = count + 1
        return x1, x2

    def dxdt(self, x1, x2, yt1, yt2):
        """
        Evaluate image velocity for a given image position x = (x1, x2)
        (yt1, yt2) = (d/dt)(y1, y2) is the source velocity vector
        """
        ih11, ih12, ih22 = self.invhessian(x1, x2)
        return ih11*yt1 + ih12*yt2, ih12*yt1 + ih22*yt2

    def dvdt(self, x1, x2, yt1, yt2):
        """
        Evaluate image acceleration for a given image position x = (x1, x2)
        (yt1, yt2) = (d/dt)(y1, y2) is the source velocity vector
        """
        ih11, ih12, ih22 = self.invhessian(x1, x2)
        h111, h112, h122, h222 = self.hessian3(x1, x2)
        v1, v2 = self.dxdt(x1, x2, yt1, yt2)
        a1 = (-(h111*ih11**2) - 2*h112*ih11*ih12 - h122*ih12**2)*v1*yt1 +  \
             (-(h112*ih11**2) - 2*h122*ih11*ih12 - h222*ih12**2)*v2*yt1 +  \
             (-(h111*ih11*ih12) - h112*ih12**2 - h112*ih11*ih22 - h122*ih12*ih22)*v1*yt2 +  \
             (-(h112*ih11*ih12) - h122*ih12**2 - h122*ih11*ih22 - h222*ih12*ih22)*v2*yt2
        a2 = (-(h111*ih11*ih12) - h112*ih12**2 - h112*ih11*ih22 - h122*ih12*ih22)*v1*yt1 +  \
             (-(h112*ih11*ih12) - h122*ih12**2 - h122*ih11*ih22 - h222*ih12*ih22)*v2*yt1 +  \
             (-(h111*ih12**2) - 2*h112*ih12*ih22 - h122*ih22**2)*v1*yt2 +  \
             (-(h112*ih12**2) - 2*h122*ih12*ih22 - h222*ih22**2)*v2*yt2
        return a1, a2

    def dnudt(self, x1, x2, yt1, yt2):
        """
        Evaluate for a given image position x = (x1, x2)
        Define a parameter nu = ln|jac|, where jac = det[hessian]
        nu is the logarithm of the (unsigned) magnification factor
        yt = (yt1, yt2) is the source velocity vector
        """
        h11, h12, h22 = self.hessian(x1, x2)
        h111, h112, h122, h222 = self.hessian3(x1, x2)
        v1, v2 = self.dxdt(x1, x2, yt1, yt2)
        jac = h11*h22 - h12**2

        # jac = h11*h22 - h12**2
        # dnu/dt = d(jac)/dt/jac
        dnudt =   (v1*(h111*h22 + h11*h122 - 2.0*h12*h112)  \
                + v2*(h112*h22 + h11*h222 - 2.0*h12*h122))/jac
        return dnudt

    def InitImgTracks(self, tini, niter=30):
        """
        Find initial image positions at a given initial epoch tini
        We assume that the initial source position is sufficiently far away from the lens
        """
        nimg = 1

        xini = np.zeros((nimg, 2), dtype=np.float64)
        # primary image far away from any of the microlenses
        y1ini, y2ini = self.getyptoft(tini)
        # external deformation matrix and Jacobian
        H11 = 1.0 - self.ka_0 - self.ga1_0
        H22 = 1.0 - self.ka_0 + self.ga1_0
        H12 =                 - self.ga2_0
        Jac = H11*H22 - H12**2
        # inverse of external deformation matrix
        iH11 =   H22/Jac
        iH22 =   H11/Jac
        iH12 = - H12/Jac
        # initial guess for the image position
        x1ini = iH11*y1ini + iH12*y2ini
        x2ini = iH12*y1ini + iH22*y2ini

        xini[0, 0], xini[0, 1] = self.EvolveImage(x1ini, x2ini, y1ini, y2ini, niter)

        return xini

    def GenImgTrackEndPoints(self):
        """
        Initial- and final-time image positions
        which are starting/ending points of image tracks
        """
        self.xini_list = self.InitImgTracks(self.tmin)
        self.xfin_list = self.InitImgTracks(self.tmax)

        self.xini_imgtrack_exist = np.array([ False for i in range(len(self.xini_list)) ])
        self.xfin_imgtrack_exist = np.array([ False for i in range(len(self.xfin_list)) ])

        return

    def SolveImageTrack(self, xi1, xi2, ti, t_arrow,  \
          max_steps=10000, t_step_eps=0.01,  \
          t_step_max=0.5, t_step_max_2=0.1, mu_cap=1e4):
        """
        Evolve an image from initial position xi = (xi1, xi2) at initial epoch ti
        An image track terminates when encountering a critical curve;
        In that case, another call is launched to start evolving a separate image  track
        """

        x_list = np.zeros((max_steps, 2), dtype=np.float64)
        t_list = np.zeros((max_steps), dtype=np.float64)
        mu_list = np.zeros((max_steps), dtype=np.float64)

        yt1, yt2 = self.yt1pt, self.yt2pt

        x10, x20 = xi1, xi2
        t0 = ti
        y10, y20 = self.getyptoft(ti)
        detj0 = self.jac(x10, x20)
        nu0 = np.log(np.abs(detj0))
        dnudt0 = self.dnudt(x10, x20, yt1, yt2)

        i = 0
        x_list[i, 0], x_list[i, 1] = x10, x20
        t_list[i] = t0
        mu_list[i] = 1.0/detj0

        t = t0
        y1, y2 = y10, y20
        x1, x2 = x10, x20
        dnudt = dnudt0
        nu = nu0
        detj = detj0

        x1_last, x2_last = x1, x2

        i = i + 1

        new_track = False

        while t >= self.tmin and t <= self.tmax and i < max_steps:

            if 1.0/np.abs(detj) > mu_cap and dnudt*t_arrow <= 0.0:

                print("Critical curve detected!")

                j1 = self.jac(x1, x2)
                j2 = self.jac(x1_last, x2_last)
                x1_jump, x2_jump = (-j1-j2)/(j1-j2)*(x1 - x1_last) + x1_last,  \
                                   (-j1-j2)/(j1-j2)*(x2 - x2_last) + x2_last

                x1_new, x2_new = self.EvolveImage(x1_jump, x2_jump, y1, y2)
                #j_new = self.jac(x1_new, x2_new)
                #if j1*j_new >= 0.0:
                #    print("FAILED JUMP!")

                t_new = t
                t_arrow_new = - t_arrow

                new_track = True

                print("Break due to critical curve")
                break

            dxdt1, dxdt2 = self.dxdt(x1, x2, yt1, yt2)
            dvdt1, dvdt2 = self.dvdt(x1, x2, yt1, yt2)

            if np.sqrt(dvdt1**2 + dvdt2**2) == 0:
                dt = t_step_max_2*t_arrow
            else:
                dt_bound = np.sqrt((dxdt1**2 + dxdt2**2)/(dvdt1**2 + dvdt2**2))
                dt = min(dt_bound*t_step_eps, t_step_max)*t_arrow

            #x_last = x # BUG: MZ commented out and replaced by line below
            x1_last, x2_last = x1, x2

            iter_count = 0
            while iter_count < 2:
                #t = t + dt
                #x1_trial, x2_trial = x1 + dxdt1*dt, x2 + dxdt2*dt

                y1_try, y2_try = self.getyptoft(t+dt)
                x1_try, x2_try = x1 + dxdt1*dt + 0.5*dvdt1*dt*dt,  \
                                 x2 + dxdt2*dt + 0.5*dvdt2*dt*dt

                x1_sol, x2_sol = self.EvolveImage(x1_try, x2_try, y1_try, y2_try)
                detj_sol = self.jac(x1_sol, x2_sol)

                if detj_sol*detj0 > 0.0: #and np.abs(np.log10(np.abs(detj_sol/detj0))) < 4.0:
                    break
                else:
                    dt = dt/10.0
                    #print(dt)

                iter_count = iter_count + 1

            y1, y2 = y1_try, y2_try
            x1, x2 = x1_sol, x2_sol
            t = t + dt
            detj = detj_sol

            nu = np.log(np.abs(detj))
            dnudt = self.dnudt(x1, x2, yt1, yt2)

            x_list[i, 0], x_list[i, 1] = x1, x2
            t_list[i] = t
            mu_list[i] = 1.0/detj

            i = i + 1

            x1_new, x2_new = x1, x2
            t_new = t
            t_arrow_new = t_arrow

        imgt = ImgTrack()

        imgt.x_list = np.zeros((i, 2), dtype=np.float64)
        imgt.x_list[:, 0], imgt.x_list[:, 1] = x_list[:i, 0], x_list[:i, 1]
        imgt.t_list = np.array([ tmp for tmp in t_list[:i] ])
        imgt.mu_list = np.array([ tmp for tmp in mu_list[:i] ])

        #print("Computation of one image track completed!")

        # Add a new image track
        if self.img_tracks is None:
            self.img_tracks = imgt
        else:
            p = self.img_tracks
            while p.nxt is not None:
                p = p.nxt
            p.set_nxt(imgt)
            imgt.set_lst(p)

        #print("Image track added!")

        return x1_new, x2_new, t_new, t_arrow_new, new_track

    def SolveAllImageTracks(self, max_steps=50000, t_step_eps=0.01, t_step_max=0.01, t_step_max_2=0.02, mu_cap=1e5):
        """
        Assume the source position is sufficiently faraway at both the initial and final times.
        Use pre-computed image positions that correspond to the initial and final source positions
        Explore all image tracks who either include those initial or final image positions as end points,
          or are continuation of such image tracks across critical curves.
        """
        eps_t = 0.1
        eps_dx = 0.01

        t_arrow = 1.0
        t_arrow_reverse = - 1.0

        track_series_count = 0

        # first follow the positive flow of time
        # start from initial-time image positions and evolve forward
        for j_ini in range(len(self.xini_list)):

            # explore if it has not been done so
            if self.xini_imgtrack_exist[j_ini]:
                continue
            else:
                self.xini_imgtrack_exist[j_ini] = True
                track_series_count = track_series_count + 1
                print('Series of image tracks # = ', track_series_count)
                #print(np.sum(self.xini_imgtrack_exist), np.sum(self.xfin_imgtrack_exist))

            x1ini, x2ini = self.xini_list[j_ini, 0], self.xini_list[j_ini, 1]
            x1new, x2new, tnew, t_arrow_new, new_track = self.SolveImageTrack(x1ini, x2ini, self.tmin, t_arrow,  \
                                          max_steps, t_step_eps, t_step_max, t_step_max_2, mu_cap)

            # iteratively continue the image track across critical curves until an end point of image position is reached
            while new_track is True:
                x1new, x2new, tnew, t_arrow_new, new_track = self.SolveImageTrack(x1new, x2new, tnew, t_arrow_new,  \
                                          max_steps, t_step_eps, t_step_max, t_step_max_2, mu_cap)

            # figure out which end point it is
            if np.abs(tnew - self.tmin) <= eps_t:
                sep = np.array([ np.sqrt((x1new - self.xini_list[i, 0])**2 + (x2new - self.xini_list[i, 1])**2) for i in range(len(self.xini_list)) ])
                ind = np.argmin(sep)
                if sep[ind] < eps_dx:
                    self.xini_imgtrack_exist[ind] = True
                else:
                    self.xini_imgtrack_exist[ind] = True
                    print('SolveAllImageTracks(): Error: abnormal end point image position', x1new, x2new, tnew)
                #print(np.sum(self.xini_imgtrack_exist), np.sum(self.xfin_imgtrack_exist))
            elif np.abs(tnew - self.tmax) <= eps_t:
                sep = np.array([ np.sqrt((x1new - self.xfin_list[i, 0])**2 + (x2new - self.xfin_list[i, 1])**2) for i in range(len(self.xfin_list)) ])
                ind = np.argmin(sep)
                if sep[ind] < eps_dx:
                    self.xfin_imgtrack_exist[ind] = True
                else:
                    self.xfin_imgtrack_exist[ind] = True
                    print('SolveAllImageTracks(): Error: abnormal end point image position', x1new, x2new, tnew)
                #print(np.sum(self.xini_imgtrack_exist), np.sum(self.xfin_imgtrack_exist))
            else:
                print('SolveAllImageTracks(): Error: abnormal end point time', x1new, x2new, tnew)

        # next follow the negative flow of time
        # start from final-time image positions and evolve backward
        for j_fin in range(len(self.xfin_list)):

            # explore if it has not been done so
            if self.xfin_imgtrack_exist[j_fin]:
                continue
            else:
                self.xfin_imgtrack_exist[j_fin] = True
                track_series_count = track_series_count + 1
                print('Series of image tracks # = ', track_series_count)
                #print(np.sum(self.xini_imgtrack_exist), np.sum(self.xfin_imgtrack_exist))

            x1fin, x2fin = self.xfin_list[j_fin, 0], self.xfin_list[j_fin, 1]
            x1new, x2new, tnew, t_arrow_new, new_track = self.SolveImageTrack(x1fin, x2fin, self.tmax, -t_arrow,  \
                                          max_steps, t_step_eps, t_step_max, t_step_max_2, mu_cap)

            # iteratively continue the image track across critical curves until an end point of image position is reached
            while new_track is True:
                x1new, x2new, tnew, t_arrow_new, new_track = self.SolveImageTrack(x1new, x2new, tnew, t_arrow_new,  \
                                          max_steps, t_step_eps, t_step_max, t_step_max_2, mu_cap)

            # figure out which end point it is
            if np.abs(tnew - self.tmin) <= eps_t:
                sep = np.array([ np.sqrt((x1new - self.xini_list[i, 0])**2 + (x2new - self.xini_list[i, 1])**2) for i in range(len(self.xini_list)) ])
                ind = np.argmin(sep)
                if sep[ind] < eps_dx:
                    self.xini_imgtrack_exist[ind] = True
                else:
                    self.xini_imgtrack_exist[ind] = True
                    print('SolveAllImageTracks(): Error: abnormal end point image position:', x1new, x2new, tnew)
                #print(np.sum(self.xini_imgtrack_exist), np.sum(self.xfin_imgtrack_exist))
            elif np.abs(tnew - self.tmax) <= eps_t:
                sep = np.array([ np.sqrt((x1new - self.xfin_list[i, 0])**2 + (x2new - self.xfin_list[i, 1])**2) for i in range(len(self.xfin_list)) ])
                ind = np.argmin(sep)
                if sep[ind] < eps_dx:
                    self.xfin_imgtrack_exist[ind] = True
                else:
                    self.xfin_imgtrack_exist[ind] = True
                    print('SolveAllImageTracks(): Error: abnormal end point image position:', x1new, x2new, tnew)
                #print(np.sum(self.xini_imgtrack_exist), np.sum(self.xfin_imgtrack_exist))
            else:
                print('SolveAllImageTracks(): Error: abnormal end point time', x1new, x2new, tnew)

        print('SolveAllImageTracks(): All successful!')

    #def

DiffractionIntegral_type.define(DiffractionIntegral.class_type.instance_type)

# --------------------------------------------------------------------------- #

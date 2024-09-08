

#Utilities to convert the time-domain delay to frequency-domain amplification factor
import numpy as np
from scipy.interpolate import griddata
from numba import *

#import diffinteg_gnfw as di
import scipy.signal.windows as win
from scipy.special import jv

#@jit
def amplification_low_w(w, y,rc, steepest_descent=True):
    '''Returns the low w approximation
       NOTE: currently adapted only to CIS!!
       following Giovanni's derivation (currently in sec IIC of draft)
       steepest_descent -> evaluate integral at z=1
    '''

    #lens, images = lens_data

    if steepest_descent==False:
        print('need to implement the integral over Bessel functions')
        return w*np.nan

    #use steepest steepest_descent
    phase = np.exp(0.25j*np.pi)

    #phi = lens.phi(phase/np.sqrt(w),0) #problem, not working for complex values
    x = phase/np.sqrt(w)
    phi = -(np.sqrt(rc**2+x**2)+rc*np.log((2*rc)/(rc+np.sqrt(rc**2+x**2))))

    integral = np.exp(-1/2.)*jv(0,phase*np.sqrt(w)*y)*(w*1j*phi + 0.5*w*phi**2)

    return 1-integral


#@jit
def compute_GO(lens_data, verbose=False, beyondGO=False):
    '''returns magnification, time delay, morse phase and image type for a list of images'''

    lens, images = lens_data

    mag = []
    delay = []
    morse_ph = []
    im_type = []
    if beyondGO:
        Delta_bGO = []

    for i, (x1,x2) in enumerate(images):

        if lens.is_regular(x1,x2):
            d11,d12,d22 = lens.ddphi(x1,x2)
        #trick to handle the central cusp in SIS
        #if d11 == 0 and d12==-1 and d22==1:
        else:
            d11, d12, d22 = np.nan, np.nan, np.nan

        trA, detA = 2.-d11-d22, (1.-d11)*(1-d22)-d12**2
        s1,  s2   = (d11-d22)/2., d12
        k         = 1.-0.5*trA
        s         = np.sqrt(s1**2+s2**2)

        if detA == 0 or np.isnan(detA):
            #print(i,'image',detA)
            mag.append(0)
        else:
            mag.append(1./detA)

        #delay.append(0.5*((x1)**2 +(x2)**2)-lens.phi(x1,x2))
        delay.append(lens.tau(x1,x2))

        if detA < 0: #saddle point
            morse_ph.append(0.5)
            im_type.append(2)
        else:
            morse_ph.append(0 if trA>0 else 1)
            im_type.append(1 if trA>0 else 3)
        #print (i, trA, detA)


            # Symmetric lens:
        if beyondGO: #NOTE: x1 != X1
            if lens.is_regular(x1,x2):
                X1 = (x1 - lens.xc1)/lens.xi0
                X2 = (x2 - lens.xc2)/lens.xi0
                R = np.sqrt(X1**2 + X2**2)
                #print(R,X1,X2, lens.is_regular(x1,x2))
                #xc = lens.rc
                dvphi  = -lens.m/lens.xi0*lens.d1phi(R)
                d2vphi = -lens.m/lens.xi0**2*lens.d2phi(R)
                d3vphi = -lens.m/lens.xi0**3*lens.d3phi(R)
                d4vphi = -lens.m/lens.xi0**4*lens.d4phi(R)
                alphai = 0.5*(1-d2vphi)
                betai  = 0.5*(1-dvphi/R)
                Dbgo   = 1/16.*(d4vphi/(2*alphai**2)+5.*d3vphi**2/(12.*alphai**3) \
                         +d3vphi/(alphai**2*R) +(alphai-betai)/(R**2*alphai*betai))
                Delta_bGO.append(Dbgo)
            else:
                Delta_bGO.append(0)


    if beyondGO:
        return [mag, delay, morse_ph, im_type, Delta_bGO]
    else:
        return [mag, delay, morse_ph, im_type]

def amplification_GO(w,
                     lens_data,
                     include_image_types = [1,2,3],
                     skip_saddles=False,
                     saddle_filter_width = 0,
                     beyondGO=False
                     ):
    ''' computes geometric optics amplification factor for a list of images
        TODO: remove skip_saddles
    '''

    if beyondGO:
        mag, delay, morse_ph, im_type, Delta_bGO = compute_GO(lens_data, beyondGO = True)
    else:
        mag, delay, morse_ph, im_type = compute_GO(lens_data, beyondGO = False)

    F_go = 0
    #print('begin')
    if skip_saddles:
        include_image_types = [n for n in include_image_types if n!=2]
    for i, mu in enumerate(mag):
        if im_type[i] not in include_image_types or mu ==0:
            #print('skip', im_type[i])
            pass
        else:
            #print('include', im_type[i])
            #geometric optics factor for each image
            if beyondGO:
                epsilon = 1e-12
                F_i = np.sqrt(np.abs(mag[i]))*(1 + 1j*Delta_bGO[i]/(w+epsilon))*np.exp(1.j*(w*delay[i] - np.pi*morse_ph[i]))
            else:
                F_i = np.sqrt(np.abs(mag[i]))*np.exp(1.j*(w*delay[i] - np.pi*morse_ph[i]))

            if im_type[i]== 2 and saddle_filter_width not in [0,np.infty]:
                #saddle point with regularized saddle_filter_width
                #print('filter width', saddle_filter_width)
                T = saddle_filter_width
                #coefficient is -w/pi (I_+ + I_-) = -w/pi*2 Re(I_+) (bc I- = I+^*)
                coef = 1j*w/np.pi*T/(1j + w*T)*(np.euler_gamma + np.log(1/T - 1j*w))
                F_i*=2*np.real(coef)

            #add amplification factors
            F_go += F_i
            #print(i)

    return F_go
#@jit()
def relative_delays(dt):
    dts = []
    for i in range(len(dt)):
        for j in range(i,len(dt)):
            diff = dt[i]-dt[j]
            if diff !=0:
                dts.append(dt[j]-dt[i])
    return dts


#TODO: move
def cis_F_wo_multigrid(y,rc,cis, ws,  grid=None, w_lims=None,window_transition=1):
    '''grid -> N tuples (f_min, f_max, w_sad)
       w_lims -> N+1
       cis -> lens class
    '''

    output = {'y': y, 'rc' : rc}

    if grid==None:
        grid = [(1e-4,100,3),
                (1e-2,1e3,0.5),
                (0.1,1.5e4,0.05)]
    if w_lims==None:
        w_lims = [1e-2, 5,100,3e4]

    window_tr = window_transition if hasattr(window_transition,'__len__') else [window_transition for i in range(len(w_lims))]
    #TODO: include tests for grids and intervals

    results = {}
    limits = {}
    wsads = {}
    #compute the amplification factor for each subgrid
    #TODO: do the contours only once!
    for i,(f_min,f_max,w_sad) in enumerate(grid):
        results[i] = cis.cis_F_wo_reg(rc, y,  f_min, f_max,
                                      window_transition=window_tr[i],
                                      saddle_filter_width=w_sad,
                                      reset_fermat_pot = True, dict_output=True)
        limits[i] = (f_min,f_max)
        wsads[i] = w_sad

    #put together the info
    w_list = [results[i]['w'] for i in range(len(results))]
    Freg_list = [results[i]['F_nonsing'] for i in range(len(results))]
    w_sads = [wsads[i] for i in range(len(results))]

    F_wo_reg_tot = F_WO_blend_domains(ws, w_list, Freg_list,
                                      w_lims,method='cubic')

    F_go_tot = F_GO_multigrid(ws, results[0]['lens_data'],
                                 w_sads,w_lims)

    #generate output
    #PUT IN DICTIONARY FORM!

    output['w'] = ws
    output['F_wo_reg'] =F_wo_reg_tot+F_go_tot
    output['F_wo'] = output['F_wo_reg']
    output['F_nonsing'] = F_wo_reg_tot
    #output['F_go'] -> this you can compute if needed
    output['F_go_reg'] = F_go_tot

    #these results are common, we can take them from any of the dictionaries
    output['lens_data'] = results[0]['lens_data']
    output['mag'], output['tau'], output['morse_phase'], output['image_type'] = compute_GO(results[0]['lens_data'])
    output['grid'] = grid
    output['w_lims'] = w_lims
    output['sad_Ts'] = w_sads

    return output

def F_WO_blend_domains(ws, w_list, Freg_list, w_lims, method='cubic', blend_factor = 0):
    ''' Blend different fourier calculations, with hard boundaries between them
    w_list, F_reg_list -> N calculations
    w_lims -> boundaries (needs N+1)
    NOTE: saddle regulator width *needs* to be common to all for continuous result, otherwise GO needs regulator
    '''
    F_wo_reg = np.zeros(len(ws),dtype='complex128')

    for i, this_w in enumerate(w_list):
        #print(i)
        thisF_wo_reg = griddata(this_w, Freg_list[i],
                                ws, method=method,
                                fill_value=0)

        low,hi = w_lims[i],w_lims[i+1]
        #This windowing avoids double counting: it also ensures that the
        window = np.where(ws>=low, 1, 0)*np.where(ws<hi,1,0)
        #print(window)
        F_wo_reg += thisF_wo_reg*window

    return F_wo_reg



def F_GO_multigrid(ws, lens_data, sad_widths, w_lims):
    ''' Blend different GO calculations, with hard boundaries between them
    sad_widths -> saddle width (N)
    w_lims -> boundaries (needs N+1)
    '''
    F_go = np.zeros(len(ws),dtype='complex128')

    #TODO: optimize by computing directly on the ws points, no need to inerpolate
    for i, this_w in enumerate(w_lims[0:-1]):
        #print(i)
        thisF_go_reg = amplification_GO(ws,lens_data,
                                        saddle_filter_width = sad_widths[i])

        low,hi = w_lims[i],w_lims[i+1]
        #This windowing avoids double counting: it also ensures that the
        window = np.where(ws>=low, 1, 0)*np.where(ws<hi,1,0)
        #print(window)
        F_go += thisF_go_reg*window

    return F_go

#@jit
def singular_part(tau,
                  lens_data,
                  verbose=False,
                  include_image_types = [1,2,3],
                  saddle_filter_width = np.infty,
                  ):
    '''singular part of \tilde I, containing the geometric optics information
       to be removed before the FFT
    '''
    mag, delay, morse_ph, im_type = compute_GO(lens_data)

    I_sing = 0*tau
    for i, typ in enumerate(im_type):
        tau_i = delay[i]
        if typ not in include_image_types:
            continue
        if typ ==2:
            #print('saddle',typ)
            saddle_filter = 1 if saddle_filter_width in [0,np.infty] else np.exp(-np.abs(tau-tau_i)/saddle_filter_width)

            I_sing += -2*np.log(np.abs(tau-tau_i))*np.sqrt(np.abs(mag[i]))*saddle_filter

        else:
            #print('nosaddle',typ)
            sign = 1 if typ==1 else -1
            #print('   ', sign)
            theta = np.heaviside(tau-tau_i,0) #np.array([1 if tau[j]>tau_i else 0 for j in range(len(tau))])

            I_sing += sign*2*np.pi*theta*np.sqrt(np.abs(mag[i]))
        if verbose:
            print( typ, tau_i, mag[i])
    return I_sing



''' TODO: split combine_contours into:
    1) construct tau_uni (N_fft = 300000,
                     extra_tau_min = 0.1,
                     tau_max_threshold = 1.01,
                     force_tau_max = np.infty,)
    2) project contours on tau_uni
    3) remove singular part?
'''

def contours_to_grid(contours,
                     tau_uni,
                     lens_data,
                     improve_interpolation = True,
                     n_fit = 3,
                     n_extra = 4,
                     eps = 1e-7,
                    ):
    ''' combines multiple contours in the time domain
        contours: list of tuples [(tau1,val1),(tau2,val2)...]
        n_fit: number of points for extrapolation near max/min
        n_extra: number of additonal points added to each contour
        eps: how close does it get to the critical point
        TODO: reconstructed saddle point depends on where the contours end. Make it symmetric by extrapolating the contours from some fiducial value, at fixed distance from the saddle.
    '''
    #1) re-interpolate all contours on a single grid

    mag, delay, morse_ph, im_type = compute_GO(lens_data)
    delay = np.array(delay)
    mag = np.array(mag)

    #initialize value grid and update it
    val_uni = tau_uni*0

    #use these values to find extrapolation points
    tau0, dtau = tau_uni[0], tau_uni[1]-tau_uni[0]

    #ideal_limits = [(delay[0],delay[1]),(delay[1],delay[2]),(delay[1],np.amax(tau_uni))]
    closest_image = []
    ideal_limits = []
    #identify the indices of the closest image
    for tau, I in contours:
        tmin, tmax  = np.amin(tau),np.amax(tau)
        d_min, d_max = np.abs(tmin-delay), np.abs(tmax - delay)
        i_min = int(np.where(d_min==np.amin(d_min))[0])
        i_max = int(np.where(d_max==np.amin(d_max))[0]) if np.amin(d_max)<1 else None

        closest_image.append((i_min,i_max))
        ideal_limits.append((delay[i_min],delay[i_max] if i_max is not None else np.infty))

    for i,(tau,val) in enumerate(contours):
        #make contours ascending
        tau, val = (tau, val) if tau[1]>tau[0] else (tau[::-1], val[::-1])

        if improve_interpolation:
            #lower end of the interval
            if (tau[0] > ideal_limits[i][0]):
                closest_image_index = closest_image[i][0]
                closest_im_type = im_type[closest_image_index]

                #add extra tau
                tx = np.linspace(delay[closest_image_index]*(1+eps),tau[0],n_extra+1)[0:-1]

                if closest_im_type in [1,3]:
                    slope,offset = np.polyfit(tau[0:n_fit],val[0:n_fit],1)
                    cx = slope *tx + offset
                elif closest_im_type in [2]:
                    tau_crit = delay[closest_image_index]
                    magnification = mag[closest_image_index]
                    cx = val[0] -2.*np.sqrt(np.abs(magnification))*(np.log(np.abs(tx-tau_crit))-np.log(np.abs(tau[0]-tau_crit)))

                tau = np.concatenate([tx,tau])
                val = np.concatenate([cx,val])

            #upper end of the interval, unless final interval
            if (np.isfinite(ideal_limits[i][1]) and tau[-1] < ideal_limits[i][1]):
                closest_image_index = closest_image[i][1]
                closest_im_type = im_type[closest_image_index]

                #add extra tau
                tx = np.linspace(tau[-1],delay[closest_image_index]*(1-eps),n_extra+1)[1:]

                if closest_im_type in [1,3]:
                    slope,offset = np.polyfit(tau[-n_fit:],val[-n_fit:],1)
                    cx = slope *tx + offset

                elif closest_im_type in [2]:
                    tau_crit = delay[closest_image_index]
                    magnification = mag[closest_image_index]
                    cx = val[-1] -2.*np.sqrt(np.abs(magnification))*(np.log(np.abs(tx-tau_crit))-np.log(np.abs(tau[-1]-tau_crit)))

                tau = np.concatenate([tau,tx])
                val = np.concatenate([val,cx])

        val_interp = scipy.interpolate.griddata(tau,val,tau_uni,fill_value=0,method='cubic')
        #plt.plot(tau,val,'--')
        val_uni+= val_interp

    return val_uni

#@jit
def project_contours(contours,
                     tau_uni,
                     lens_data,
                     saddle_filter_width = np.infty,
                     improve_interpolation = True,
                     verbose=False):
    ''' combines multiple contours in the time domain
        if needed subtracts the singular part
        contours: list of tuples [(tau1,val1),(tau2,val2)...] -> TODO: order is funny, need to make consistent
    '''

    val_uni = contours_to_grid(contours,tau_uni,lens_data,improve_interpolation = improve_interpolation)

    #compute and decompose singular part
    I_sing = singular_part(tau_uni, lens_data, include_image_types=[1,2,3], saddle_filter_width=saddle_filter_width)
    I_extrem = singular_part(tau_uni, lens_data, include_image_types=[1,3])
    I_sad = I_sing - I_extrem

    return tau_uni, val_uni, I_sing, I_extrem, I_sad



def tau_range(f_min,
              f_max,
              lens_data=None,
              window_transition = 0.2,
              tau_min_extend = 0.1,
              N_fft_power_2 = False):

    if lens_data != None:
        mag, delay, morse_ph, im_type = compute_GO(lens_data)
        td_min = min(delay)
    else:
        td_min = 0

    Dtau = 1/f_min #total width

    N_fft = 2**int(np.log2(f_max/f_min)) if N_fft_power_2 else int(f_max/f_min) #FFT faster for the right choices of N_fft

    #make sure that the window function operates only
    tau_min = min(td_min, td_min-window_transition/2.*Dtau) - tau_min_extend

    #TODO: the padding is shifted when tau_max is reduced because of interpolation
    tau_max = Dtau + tau_min

    return [tau_min,tau_max,N_fft]

def f2tau_win(f_min,
              f_max,
              contours,
              lens_data,
              window_transition = 0.2,
              tau_min_extend = 0.1,
              N_fft_power_2 = False
             ):
    '''Get tau grid from min/max frequency and window function
       make sure that the window does not conflict with the signal
       NOTE:the frequencies are typically larger than what's asked for
    '''

    tau_min, tau_max,N_fft = tau_range(f_min, f_max, lens_data,window_transition,
    tau_min_extend, N_fft_power_2)


    max_tau_from_contours = max([np.amax(c[0]) for c in contours])

    if tau_max > max_tau_from_contours:
        print('wished tau_max=%g larger than available for interpolation (%g)'%(tau_max,max_tau_from_contours))
        tau_max = max_tau_from_contours

    tau_uni = np.linspace(tau_min,tau_max, N_fft)
    window = win.tukey(N_fft,alpha=window_transition)


    return tau_uni, window

def FFT_2_F(tau_uni, val_uni, window = 1, verbose=False):
    '''computes the FFT of a quantity, no GO neded'''
    tau_max,tau_min = np.amax(tau_uni), np.amin(tau_uni)
    #mag, delay, morse_ph, im_type = compute_GO(lens_data)
    #tau_L = np.amin(delay)
    tau_L = 0

    N_fft = len(tau_uni)
    dtau = (tau_max-tau_min)/N_fft
    f_fft = np.fft.rfftfreq(N_fft,dtau)
    w = f_fft*2*np.pi


    if verbose == True:
        print('tau_L =', tau_L)
        print('tau_min = %g, tau_max = %g'%(tau_min,tau_max))

    #NOTE: DFT things t=0 corresponds to the first point of tilde I_n
    #need to correct with a phase factor to recover the GO result
    time_shift = np.exp(-1j*w*tau_min)


    F_w = -1j*f_fft*np.conj(np.fft.rfft(val_uni*window)*time_shift)*dtau

    return w, F_w

def FFT_2_F_WO(tau_uni,
               val_uni,
               lens_data,
               #multiple_inputs = False,
               window=1,
               saddle_filter_width = np.infty,
               verbose = False,
               ):
    '''compute the FFT to the frequency domain
       TODO: allow for multiple_inputs
       the output includes the GO and its different contributions
    '''

    #0) get important quantities

    tau_max,tau_min = np.amax(tau_uni), np.amin(tau_uni)
    mag, delay, morse_ph, im_type = compute_GO(lens_data)
    tau_L = np.amin(delay)

    N_fft = len(tau_uni)
    dtau = (tau_max-tau_min)/N_fft
    f_fft = np.fft.rfftfreq(N_fft,dtau)
    w = f_fft*2*np.pi


    if verbose == True:
        print('tau_L =', tau_L)
        print('tau_min = %g, tau_max = %g'%(tau_min,tau_max))

    #NOTE: DFT things t=0 corresponds to the first point of tilde I_n
    #need to correct with a phase factor to recover the GO result
    time_shift = np.exp(-1j*w*tau_min)


    F_wo = -1j*f_fft*np.conj(np.fft.rfft(val_uni*window)*time_shift)*dtau

    F_go = amplification_GO(w,lens_data,skip_saddles=False,saddle_filter_width=saddle_filter_width)
    F_go_extrem = amplification_GO(w,lens_data,skip_saddles=True)
    F_go_sad = F_go - F_go_extrem

    #print(len(F_wo))

    return w, F_wo, F_go, F_go_extrem, F_go_sad


def new_singular_part(tau, dIdpar,dt,dt_param,epsilon_D_I=1e-4):
    ''' compute the discontinuities for fisher derivatives at critical points
        dIdpar -> dI/dx from finite differences
        dt ->  dt[x] critical points at fiducial
        dt_param -> dt[x+dx], needed to know the sign of the steps
        returns dI_sing, dF_sing, steps
    '''


    Npoints, tmin, tmax = len(tau), tau[0], tau[-1]
    dtau = (tmax-tmin)/Npoints

    #difference in time delays
    deltaDt = np.array(dt)-np.array(dt_param)

    if np.any(deltaDt>dtau):
        print('differnce in Fermat pot. between fidutial & variation larger than grid: DeltaT/dtau =',(deltaDt/dtau))

    #points closest to the maxima/minima
    N_for_dt = []
    for i,x in enumerate(dt):
        N=(x-tmin)/dtau
        #print(i,N)
        N_for_dt.append(int(N))


    new_sing_part = np.zeros(Npoints)

    discontinuities = []
    #look around those points
    for i,N in enumerate(N_for_dt):
        D_forw, D_back = dIdpar[N+1]-dIdpar[N], dIdpar[N]-dIdpar[N-1]
        #print(i, D_forw,D_back)
        if max(np.abs(D_forw),np.abs(D_back))<epsilon_D_I: #previously: np.abs(deltaDt[i])<1e-10: -> skip the minima, but important for WL
            discontinuities.append(0)
            continue
        #print(i, tau[N]<dt[i])
        if tau[N]<dt[i]:
            DeltaI = D_forw if np.abs(D_forw/D_back) > 1 else 0
            #print(i,'forw',DeltaI)
        elif tau[N]>dt[i]:
            DeltaI = D_back if np.abs(D_back/D_forw) > 1 else 0
            #print(i, 'back',DeltaI)
        else:
            print('error!')

        discontinuities.append(DeltaI)

        #check we get the largest dicontinuity
        #print('%i, %.4g,%.4g,%.4g'%(i,DeltaI,(D_forw),(D_back)))

        new_sing_part += DeltaI*np.heaviside((tau-dt[i]),0.5)

    #print(dt, DI)
    return new_sing_part, discontinuities

dFdrc_count = 0

def new_singular_counterpart(w, dt, steps_param):
    ''' compute the F_sing counterpart for the fisher derivs
        w -> dimensionless sample_frequencies
        dt -> time delays
        steps_param -> discontinuities in the fisher deriv
    '''
    #compute counterparts for each step
    dFdparam = np.complex(0)
    for i, step in enumerate(steps_param):
        #delay = (dt[i]+dtc[i])/2 -> maybe this would improve
        delay = dt[i]
        #NOTE: there is no morse phase for this term!
        dFdparam += step/(2*np.pi)*np.exp(1j*(w*delay))
    return dFdparam

def compute_amplification(f_min,
                          f_max,
                          contours,
                          lens_data,
                          N_fft_power_2 = True,
                          window_transition=0.7,
                          tau_min_extend=0,
                          saddle_filter_width=np.infty,
                          improve_interpolation=True,
                          verbose=False,
                          full_output = False,
                          ):
    ''' Computes amplification factor via FFT
        TODO: fix saddle saddle_filter
    '''

    #compute the uniform grid and window function
    tau_uni, window = f2tau_win(f_min,
                                f_max,
                                contours,
                                lens_data,
                                tau_min_extend=tau_min_extend,
                                window_transition=window_transition,
                                N_fft_power_2=N_fft_power_2
                                )


    #project contours onto uniform grid, taking care of interpolations
    time_domain = project_contours(contours,
                                   tau_uni,
                                   lens_data,
                                   saddle_filter_width = saddle_filter_width,
                                   improve_interpolation = improve_interpolation,
                                   verbose=verbose)

    tau_uni, I_full, I_sing, I_extrem, I_sad = time_domain

    I_regular = I_full - I_sing

    #fourier transform full I(tau)
    #TODO: not really needed!
    #important to ignore the saddle filter!
    freq_domain = FFT_2_F_WO(tau_uni,
                                  I_full,
                                  lens_data,
                                  window=window,
                                  saddle_filter_width=np.infty,
                                  verbose=verbose)

    w, F_wo, F_go, F_go_extrem, F_go_sad = freq_domain

    #fourier transform regular I(tau)
    freq_domain_reg = FFT_2_F_WO(tau_uni,
                                  I_regular,
                                  lens_data,
                                  window=window,
                                  saddle_filter_width=saddle_filter_width,
                                  verbose=verbose)

    w, F_nonsing, F_go_reg, F_go_extrem_reg, F_go_sad_reg = freq_domain_reg


    F_wo_reg = F_nonsing + F_go_reg

    if full_output:
        return w, F_wo, F_wo_reg, F_go, F_go_reg,  time_domain, freq_domain, freq_domain_reg, window
    else:
        return w, F_wo, F_wo_reg, F_go, F_go_reg


import scipy

#@jit(i8(types.Array(f8, 1, 'A', readonly=True), f8, i8))
#@jit
def compute_envelopes(s):
    ''' returns the upper and lower envelope of a data vector'''
    #s = np.array(s)
    num = len(s)
    q_u = np.zeros(num)
    q_l = np.zeros(num)

    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

    u_x = [0,]
    u_y = [s[0],]

    l_x = [0,]
    l_y = [s[0],]

    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in np.arange(1,len(s)-1):
        if (np.sign(s[k]-s[k-1])==1) and (np.sign(s[k]-s[k+1])==1):
            u_x.append(k)
            u_y.append(s[k])

        if (np.sign(s[k]-s[k-1])==-1) and ((np.sign(s[k]-s[k+1]))==-1):
            l_x.append(k)
            l_y.append(s[k])

    #Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

    u_x.append(len(s)-1)
    u_y.append(s[-1])

    l_x.append(len(s)-1)
    l_y.append(s[-1])

    #Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

    u_p = scipy.interpolate.interp1d(u_x,u_y, kind = 'linear',bounds_error = False, fill_value=0.0)
    l_p = scipy.interpolate.interp1d(l_x,l_y,kind = 'linear',bounds_error = False, fill_value=0.0)

    #Evaluate each model over the domain of (s)
    for k in np.arange(0,len(s)):
        q_u[k] = u_p(k)
        q_l[k] = l_p(k)

    return q_l, q_u




#@jit
def project_contours_old(contours,
                     tau_uni,
                     lens_data,
                     saddle_filter_width = np.infty,
                     improve_interpolation = True,
                     verbose=False):
    ''' combines multiple contours in the time domain
        if needed subtracts the singular part
        contours: list of tuples [(tau1,val1),(tau2,val2)...] -> TODO: order is funny, need to make consistent
    '''

    #1) re-interpolate all contours on a single grid

    mag, delay, morse_ph, im_type = compute_GO(lens_data)
    delay = np.array(delay)
    #initialize value grid and update it
    val_uni = tau_uni*0

    #use these values to find extrapolation points
    tau0, dtau = tau_uni[0], tau_uni[1]-tau_uni[0]

    #fill in the correct values after the interpolation
    #ideal_limits = [(delay[0],delay[1]),(delay[1],delay[2]),(delay[1],np.amax(tau_uni))] #limits for each contour TODO: automatize by comparing with critical tau's
    ideal_limits = [(delay[i], delay[i+1]) if i+1<len(delay) else (delay[i],np.amax(tau_uni)) for i,dt in enumerate(delay)]
    #identify the indices of the closest image
    closest_image = []
    for tmin, tmax in ideal_limits:
        d_min, d_max = np.abs(tmin-delay), np.abs(tmax - delay)
        i_min = (d_min==np.amin(d_min)).nonzero()[0]
        i_max = (d_max==np.amin(d_max)).nonzero()[0] if tmax != np.amax(tau_uni) else None #slows down but negligible
        closest_image.append([i_min,i_max])

        #closest_image[-1][1] = None #this assumes the last contour is the asymptotic one

    #TODO: make the association tied to the contour
    #for min, max in ideal_limits:

    #TODO: fix saddle point extrapolation
    saddle_extrap_factor = np.ones(len(tau_uni)) #zero for points where extrapolation is neded
    saddle_extrap_points = []
    saddle_which_one = [] #keep track of which saddle point!

    for i,(tau,val) in enumerate(contours):
        #make contours ascending
        tau, val = (tau, val) if tau[1]>tau[0] else (tau[::-1], val[::-1])

        #interpolate
        val_interp = scipy.interpolate.griddata(tau,val,tau_uni,fill_value=0,method='cubic')

        if improve_interpolation:
            #NOTE: could interpolate the saddle points here by filling in 1/4 or 1/2 of the total singular contribution depending on whether the contour touches 1 or 2 of the saddle lobes. Would require doing the computation in more detail
            #find the indices of points where interpolation is needed
            low_interp_j = ((tau_uni>ideal_limits[i][0])*(tau_uni<tau[0])).nonzero()[0]
            high_interp_j = ((tau_uni<ideal_limits[i][1])*(tau_uni>tau[-1])).nonzero()[0]

            interp_j = np.concatenate([low_interp_j,high_interp_j])

            if len(interp_j) > 0:
                val_interp_near = scipy.interpolate.griddata(tau,val, tau_uni[interp_j],method='nearest')
                if verbose:
                    print('contour %i: found %i points that need improved interpolation'%(i,len(interp_j)))
            for j_pos, j in enumerate(interp_j):

                closest_val = val_interp_near[j_pos]
                closest_tau = tau[0] if tau_uni[j]<tau[0] else tau[-1]
                #print(closest_image[i][0],closest_image[i][1])
                closest_image_index = closest_image[i][0] if tau_uni[j]<tau[0] else closest_image[i][1]
                if len(closest_image_index) == 0:
                    continue
                #print(closest_image_index)
                closest_im_type = im_type[int(closest_image_index)]
                #print(closest_im_type)
                if closest_im_type in [1,3]:
                    #maximum/minimum: only one region contributes to a max/min
                    val_interp[j] = closest_val
                elif closest_im_type in [2]:
                    #take note of a saddle point & correct later (bc saddle point can pertain to multiple contours
                    #val_interp[j] = closest_val -2.*np.sqrt(np.abs(magnification))*(np.log(np.abs(taux-tau_crit))-np.log(np.abs(closest_tau-tau_crit)))
                    saddle_extrap_factor[j] = 0
                    if j not in saddle_extrap_points:
                        saddle_extrap_points.append(j)
                        saddle_which_one.append(closest_image_index[0])
                           #find the points where
                else:
                    pass

        val_uni += val_interp

    if len(saddle_extrap_points) == 0 and verbose == True:
            print('no ponits in saddle_extrap_points')
    if improve_interpolation and len(saddle_extrap_points) != 0:
        #error if several saddle points!
        if len(set(saddle_which_one)) > 1:
            print ('Extrapolation needed near multiple saddle points, need to revisit this part of the code')

        #now extrapolate around the saddle point (ASSUMES JUST ONE!)
        #TODO: in a more general setting this should be split over the different saddle points!
        tI_closest_below, tI_closest_above = val_uni[min(saddle_extrap_points)-1],val_uni[max(saddle_extrap_points)+1]
        tau_closest_below, tau_closest_above = tau_uni[min(saddle_extrap_points)-1],tau_uni[max(saddle_extrap_points)+1]
        for i,n in enumerate(saddle_extrap_points):
            taux = tau_uni[n]
            tau_crit = delay[saddle_which_one[i]]
            magnification = mag[saddle_which_one[i]]

            closest_val = tI_closest_below if taux < tau_crit else tI_closest_above
            closest_tau = tau_closest_below if taux < tau_crit else tau_closest_above

            val_uni[n] = closest_val -2.*np.sqrt(np.abs(magnification))*(np.log(np.abs(taux-tau_crit))-np.log(np.abs(closest_tau-tau_crit)))

    #compute and decompose singular part
    I_sing = singular_part(tau_uni, lens_data,include_image_types=[1,2,3],saddle_filter_width=saddle_filter_width)
    I_extrem = singular_part(tau_uni, lens_data,include_image_types=[1,3])
    I_sad = I_sing - I_extrem

    return tau_uni, val_uni, I_sing, I_extrem, I_sad

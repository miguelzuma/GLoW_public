#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_ellint.h>

#include "common.h"
#include "analytic_SIS_lib.h"
#include "special_lib.h"

#define EPS_SOFT pprec.as_eps_soft

// =================================================================

double I_func(double a, double b, double c, double d, gsl_mode_t prec_mode)
{
    double It;
    double n, r, C;
    double soft;
    double K, Pi;

    soft = EPS_SOFT;

    n = (a-b)/(a-c + soft);
    r = sqrt(n*(c-d)/(b-d + soft));
    C = 2/sqrt((a-c)*(b-d + soft));

    K = gsl_sf_ellint_Kcomp(r, prec_mode);
    Pi = gsl_sf_ellint_Pcomp(r, -n, prec_mode);

    It = C*((b-c)*Pi + c*K);

    return It;
}

// options include GSL_PREC_DOUBLE, GSL_PREC_SINGLE, GSL_PREC_APPROX
double It_SIS_VarPrec(double tau, double y, double psi0, gsl_mode_t prec_mode)
{
    double tau12, tau23;
    double u, R, sqr;
    double It;

    R = (psi0-y)/(psi0+y);

    // Step
    if(tau < 0)
        It = 0;
    else
    {
        u = sqrt(2*tau)/(psi0+y);
        tau12 = 0.5*(psi0+y)*(psi0+y);
        tau23 = (1-R*R)*tau12;

        // Region 1
        if(tau > tau12)
        {
            sqr = sqrt(u*u + R*R - 1);
            It = 4*I_func(1+u, R+sqr, 1-u, R-sqr, prec_mode);
        }

        // Region 2
        if( (tau < tau12) && (tau > tau23) )
        {
            sqr = sqrt(1 - u*u);

            // Region 2A
            if(R > 0)
                It = 4*I_func(1, R, sqr, -sqr, prec_mode);

            //Region 2B
            else
                It = 4*I_func(1, sqr, -sqr, R, prec_mode);
        }

        // Region 3
        if(tau < tau23)
        {
            sqr = sqrt(1 - u*u);
            It = 4*I_func(1, sqr, R, -sqr, prec_mode);
        }
    }

    return It;
}

double It_SIS_DoublePrec(double tau, double y, double psi0)
{
    return It_SIS_VarPrec(tau, y, psi0, GSL_PREC_DOUBLE);
}

double It_SIS_SinglePrec(double tau, double y, double psi0)
{
    return It_SIS_VarPrec(tau, y, psi0, GSL_PREC_SINGLE);
}

double It_SIS_ApproxPrec(double tau, double y, double psi0)
{
    return It_SIS_VarPrec(tau, y, psi0, GSL_PREC_APPROX);
}


// =================================================================

double integrand_f0_Direct(double theta, void *param)
{
    double alpha, alpha0, r;
    double *p = (double *)param;

    alpha0 = p[0];
    r = p[1];
    alpha = alpha0*(1 + r*cos(theta));

    return alpha*f_fresnel(-alpha);
}

double integrand_g0_Direct(double theta, void *param)
{
    double alpha, alpha0, r;
    double *p = (double *)param;

    alpha0 = p[0];
    r = p[1];
    alpha = alpha0*(1 + r*cos(theta));

    return alpha*g_fresnel(-alpha);
}

double integrand_If_Direct(double theta, void *param)
{
    double alpha, alpha0, r, sign;
    double *p = (double *)param;

    alpha0 = p[0];
    r = p[1];
    sign = p[2]; // either +1 or -1
    alpha = alpha0*(sign + r*cos(theta));

    //HVR_DEBUG
    //~ if(alpha < 0)
        //~ printf("alpha=%e    sign=%g\n", alpha, sign);

    return alpha*f_fresnel(alpha);
}

double integrand_Ig_Direct(double theta, void *param)
{
    double alpha, alpha0, r, sign;
    double *p = (double *)param;

    alpha0 = p[0];
    r = p[1];
    sign = p[2]; // either +1 or -1
    alpha = alpha0*(sign + r*cos(theta));

    return alpha*g_fresnel(alpha);
}

double integrand_Is_Direct(double theta, void *param)
{
    double alpha, alpha0, r, sign;
    double *p = (double *)param;

    alpha0 = p[0];
    r = p[1];
    sign = p[2]; // either +1 or -1
    alpha = alpha0*(sign + r*cos(theta));

    return alpha*sin(M_PI_2*alpha*alpha);
}

double integrand_Ic_Direct(double theta, void *param)
{
    double alpha, alpha0, r, sign;
    double *p = (double *)param;

    alpha0 = p[0];
    r = p[1];
    sign = p[2]; // either +1 or -1
    alpha = alpha0*(sign + r*cos(theta));

    return alpha*cos(M_PI_2*alpha*alpha);
}

double integrand_Isc_OscWeight(double u, void *param)
{
    // integrate with sin(wu) or cos(wu) weighting
    double a, b;
    double *p = (double *)param;
    double sqrt_u = sqrt(u);

    a = p[0];
    b = p[1];

    return 1./sqrt((b-sqrt_u)*(sqrt_u-a));
}

double integrand_Jsc_OscWeight(double u, void *param)
{
    // integrate with sin(wu) or cos(wu) weighting
    double a, b;
    double *p = (double *)param;
    double sqrt_u = sqrt(u);

    a = p[0];
    b = p[1];

    return 1./sqrt((b-sqrt_u)*(sqrt_u+a));
}


double complex Fw_SIS(double w, double y, double psi0, int method)
{
    int n, n_max, n_max_switch;
    double alpha0, r, Delta;
    double complex phase_shift, Fw;

    alpha0 = psi0*sqrt(w/M_PI);
    r = y/psi0;

    if(method == sis_direct)
        Fw = Fw_SIS_direct(alpha0, r);

    if(method == sis_osc)
    {
        // half of the total number of oscillations in the integrand
        n_max = (int)(r*r*w/M_PI);

        // when n_max exceeds this, switch to osc method
        n_max_switch = pprec.as_FwSIS_nmax_switch;

        // number of oscillations at the limits of the integral to still
        // be integrated with the direct method (avoid alg divergence)
        n = pprec.as_FwSIS_n;
        Delta = sqrt(M_PI*n/w);

        if(n_max < n_max_switch)
            Fw = Fw_SIS_direct(alpha0, r);
        else
        {
            if(r < 1)    // strong lensing
                Fw = Fw_SIS_SL_osc(alpha0, r, Delta);
            else
                Fw = Fw_SIS_WL_osc(alpha0, r, Delta);
        }
    }

    // add scaling with tmin and y^2/2 before exiting
    // i.e. multiply by exp(i*w*0.5*(psi0+y)**2)
    phase_shift = cexp(I*0.5*w*(psi0+y)*(psi0+y));
    Fw *= phase_shift;

    return Fw;
}

double complex Fw_SIS_direct(double alpha0, double r)
{
    int sub_int_limit;
    double p[2];
    double error, If, Ig;
    double atol, rtol, xmin, xmax;
    double complex Fw;
    gsl_integration_workspace *wk;
    gsl_function Ff, Fg;

    sub_int_limit = pprec.as_FwDirect.n;
    wk = gsl_integration_workspace_alloc(sub_int_limit);

    p[0] = alpha0;
    p[1] = r;
    Ff.function = integrand_f0_Direct;
    Fg.function = integrand_g0_Direct;
    Ff.params = p;
    Fg.params = p;

    xmin = 0;
    xmax = M_PI;

    rtol = pprec.as_FwDirect.epsrel;
    atol = pprec.as_FwDirect.epsabs;

    gsl_integration_qags(&Ff, xmin, xmax, atol, rtol, sub_int_limit,
                         wk, &If, &error);
    gsl_integration_qags(&Fg, xmin, xmax, atol, rtol, sub_int_limit,
                                  wk, &Ig, &error);

    Fw = 1 + If - I*Ig;

    gsl_integration_workspace_free(wk);

    return Fw;
}

double complex Fw_SIS_SL_osc(double alpha0, double r, double Delta)
{
    int n_levels, sub_int_limit;
    double p[3];
    double error, xmin, xmax;
    double atol, rtol, th_a, th_b, a, b;
    double If, Ig, Is, Ic, Is_tmp, Ic_tmp;
    double complex Fw;
    gsl_function F;
    gsl_integration_workspace *wk;
    gsl_integration_qawo_table *t_osc;

    n_levels = pprec.as_slFwOsc_Osc.n;
    sub_int_limit = pprec.as_slFwOsc_Direct.n;
    wk = gsl_integration_workspace_alloc(sub_int_limit);

    // ---------   DIRECT PART
    // ------------------------------------------
    rtol = pprec.as_slFwOsc_Direct.epsrel;
    atol = pprec.as_slFwOsc_Direct.epsabs;

    p[0] = alpha0;
    p[1] = r;
    p[2] = 1;
    F.params = p;

    th_a = acos(-1 + Delta/r);
    th_b = acos(1 - Delta/r);

    // f and g
    F.function = integrand_If_Direct;
    gsl_integration_qags(&F, 0, M_PI, atol, rtol, sub_int_limit, wk,
                                  &If, &error);
    F.function = integrand_Ig_Direct;
    gsl_integration_qags(&F, 0, M_PI, atol, rtol, sub_int_limit, wk,
                                  &Ig, &error);

    // sine (direct)
    F.function = integrand_Is_Direct;
    gsl_integration_qags(&F, 0, th_b, atol, rtol, sub_int_limit, wk,
                         &Is, &error);
    gsl_integration_qags(&F, th_a, M_PI, atol, rtol, sub_int_limit, wk,
                                  &Is_tmp, &error);
    Is += Is_tmp;

    // cosine (direct)
    F.function = integrand_Ic_Direct;
    gsl_integration_qags(&F, 0, th_b, atol, rtol, sub_int_limit, wk,
                         &Ic, &error);
    gsl_integration_qags(&F, th_a, M_PI, atol, rtol, sub_int_limit, wk,
                                  &Ic_tmp, &error);
    Ic += Ic_tmp;


    // ---------   OSCILLATING PART
    // ------------------------------------------
    rtol = pprec.as_slFwOsc_Osc.epsrel;
    atol = pprec.as_slFwOsc_Osc.epsabs;

    a = 1-r;
    b = 1+r;

    p[0] = a;
    p[1] = b;
    F.params = p;

    F.function = integrand_Isc_OscWeight;

    xmin = gsl_pow_2(a+Delta);
    xmax = gsl_pow_2(b-Delta);

    // setting table for the sine
    t_osc = gsl_integration_qawo_table_alloc(M_PI_2*alpha0*alpha0, xmax-xmin,
                                             GSL_INTEG_SINE, n_levels);
    gsl_integration_qawo(&F, xmin, atol, rtol, sub_int_limit, wk, t_osc,
                                  &Is_tmp, &error);
    Is += 0.5*alpha0*Is_tmp;

    // change the table now to integrate the cosine
    gsl_integration_qawo_table_set(t_osc, M_PI_2*alpha0*alpha0, xmax-xmin,
                                   GSL_INTEG_COSINE);
    gsl_integration_qawo(&F, xmin, atol, rtol, sub_int_limit, wk, t_osc,
                                  &Ic_tmp, &error);
    Ic += 0.5*alpha0*Ic_tmp;


    // ---------   FINAL RESULT
    // ------------------------------------------
    Ig = -Ig + Is + Ic;
    If = -If - Is + Ic;

    Fw = 1 + If - I*Ig;

    // clean
    gsl_integration_qawo_table_free(t_osc);
    gsl_integration_workspace_free(wk);

    return Fw;
}

double complex Fw_SIS_WL_osc(double alpha0, double r, double Delta)
{
    int n_levels, sub_int_limit;
    double p[3];
    double error, xmin, xmax;
    double atol, rtol, th_0, th_b, a, b;
    double If, Ig, If_tmp, Ig_tmp;
    double Js, Jc, Js_tmp, Jc_tmp;
    double complex Fw;
    gsl_function F;
    gsl_integration_workspace *wk;
    gsl_integration_qawo_table *t_osc;

    n_levels = pprec.as_wlFwOsc_Osc.n;
    sub_int_limit = pprec.as_wlFwOsc_Direct.n;
    wk = gsl_integration_workspace_alloc(sub_int_limit);


    // ---------   DIRECT PART
    // ------------------------------------------
    rtol = pprec.as_wlFwOsc_Direct.epsrel;
    atol = pprec.as_wlFwOsc_Direct.epsabs;

    p[0] = alpha0;
    p[1] = r;
    F.params = p;

    th_b = acos(1 - Delta/r);

    // Positive contribution first
    p[2] = 1;
    th_0 = acos(-1./r);

    F.function = integrand_If_Direct;
    gsl_integration_qags(&F, 0, th_0, atol, rtol, sub_int_limit, wk,
                                  &If, &error);
    F.function = integrand_Ig_Direct;
    gsl_integration_qags(&F, 0, th_0, atol, rtol, sub_int_limit, wk,
                                  &Ig, &error);
    F.function = integrand_Is_Direct;
    gsl_integration_qags(&F, 0, th_b, atol, rtol, sub_int_limit, wk,
                                  &Js, &error);
    F.function = integrand_Ic_Direct;
    gsl_integration_qags(&F, 0, th_b, atol, rtol, sub_int_limit, wk,
                                  &Jc, &error);

    // Negative contribution
    p[2] = -1;
    th_0 = acos(1./r);

    F.function = integrand_If_Direct;
    gsl_integration_qags(&F, 0, th_0, atol, rtol, sub_int_limit, wk,
                                  &If_tmp, &error);
    F.function = integrand_Ig_Direct;
    gsl_integration_qags(&F, 0, th_0, atol, rtol, sub_int_limit, wk,
                                  &Ig_tmp, &error);

    If += If_tmp;
    Ig += Ig_tmp;


    // ---------   OSCILLATING PART
    // ------------------------------------------
    rtol = pprec.as_wlFwOsc_Osc.epsrel;
    atol = pprec.as_wlFwOsc_Osc.epsabs;

    a = r-1;
    b = r+1;

    p[0] = a;
    p[1] = b;
    F.params = p;

    F.function = integrand_Jsc_OscWeight;

    xmin = 0;
    xmax = gsl_pow_2(b-Delta);

    // setting table for the sine
    t_osc = gsl_integration_qawo_table_alloc(M_PI_2*alpha0*alpha0, xmax-xmin,
                                             GSL_INTEG_SINE, n_levels);
    gsl_integration_qawo(&F, xmin, atol, rtol, sub_int_limit, wk, t_osc,
                                  &Js_tmp, &error);
    Js += 0.5*alpha0*Js_tmp;

    // change the table now to integrate the cosine
    gsl_integration_qawo_table_set(t_osc, M_PI_2*alpha0*alpha0, xmax-xmin,
                                   GSL_INTEG_COSINE);
    gsl_integration_qawo(&F, xmin, atol, rtol, sub_int_limit, wk, t_osc,
                                  &Jc_tmp, &error);
    Jc += 0.5*alpha0*Jc_tmp;


    // ---------   FINAL RESULT
    // ------------------------------------------
    Ig = -Ig + Js + Jc;
    If = -If - Js + Jc;

    Fw = 1 + If - I*Ig;

    // clean
    gsl_integration_qawo_table_free(t_osc);
    gsl_integration_workspace_free(wk);

    return Fw;
}

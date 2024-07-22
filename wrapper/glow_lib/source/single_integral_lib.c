#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>

#include "common.h"
#include "lenses_lib.h"
#include "roots_lib.h"
#include "single_integral_lib.h"

#define EPS_SOFT 0

static const int keys_gauss[m_integral_g61+1] = {GSL_INTEG_GAUSS15, GSL_INTEG_GAUSS21, GSL_INTEG_GAUSS31,
                                                 GSL_INTEG_GAUSS41, GSL_INTEG_GAUSS51, GSL_INTEG_GAUSS61};


// =================================================================

// auxiliary function to be used with qsort
int compare_double(const void *a, const void *b)
{
    // negative -> a before b
    // positive -> b before a
    // 0 -> equal
    double x, y;

    x = *(double *)a;
    y = *(double *)b;

    if(x > y)
        return 1;
    else if(x < y)
        return -1;

    return 0;
}

// check that the crit points are ordered in growing x1
int check_sorting(int n_points, CritPoint *points)
{
    int i, has_correct_order = _TRUE_;

    for(i=0;i<n_points-1;i++)
        if(points[i].x1 > points[i+1].x1)
            has_correct_order = _FALSE_;

    return has_correct_order;
}


// ======  operate with brackets
// =================================================================

void display_Bracket(Bracket *br)
{
    printf(" - Bracket info:\n");
    printf("   * a = %e\n", br->a);
    printf("   * b = %e\n", br->b);
}

double find_root_Bracket(double xguess, pSIntegral *p)
{
    int status;
    int iter, max_iter;
    double epsabs, epsrel;
    double x, x0;

    const gsl_root_fdfsolver_type *T;
    gsl_root_fdfsolver *s;
    gsl_function_fdf FDF;

    FDF.f = phi_SingleIntegral;
    FDF.df = dphi_SingleIntegral;
    FDF.fdf = phi_dphi_SingleIntegral;
    FDF.params = p;

    max_iter = pprec.si_findRootBracket.max_iter;
    epsabs   = pprec.si_findRootBracket.epsabs;
    epsrel   = pprec.si_findRootBracket.epsrel;
    T = get_fdfRoot(pprec.si_findRootBracket.id);

    x = xguess;
    s = gsl_root_fdfsolver_alloc(T);
    gsl_root_fdfsolver_set(s, &FDF, x);

    iter = 0;
    do
    {
        status = gsl_root_fdfsolver_iterate(s);
        x0 = x;
        x = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x, x0, epsabs, epsrel);

        // HVR_DEBUG
        //~ if(status == GSL_SUCCESS)
            //~ printf("Converged: x=%e    iter=%d/%d\n", x, iter, max_iter);

        iter++;
    }
    while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fdfsolver_free(s);

    return x;
}

double find_bracket_Bracket(double a, double b, pSIntegral *p)
{
    int status;
    int iter, max_iter;
    double epsabs, epsrel;
    double x, x_lo, x_hi;

    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    gsl_function F;

    F.function = phi_SingleIntegral;
    F.params = p;

    max_iter = pprec.si_findBrackBracket.max_iter;
    epsabs = pprec.si_findBrackBracket.epsabs;
    epsrel = pprec.si_findBrackBracket.epsrel;
    T = get_fRoot(pprec.si_findBrackBracket.id);

    s = gsl_root_fsolver_alloc(T);

    if(a > b)
    {
        x_lo = b;
        x_hi = a;
    }
    else
    {
        x_lo = a;
        x_hi = b;
    }

    gsl_root_fsolver_set(s, &F, x_lo, x_hi);

    iter = 0;
    do
    {
        iter++;
        status = gsl_root_fsolver_iterate(s);
        x = gsl_root_fsolver_root(s);
        x_lo = gsl_root_fsolver_x_lower(s);
        x_hi = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lo, x_hi, epsabs, epsrel);

        // HVR_DEBUG
        //~ if(status == GSL_SUCCESS)
            //~ printf("Converged: x_lo=%e  x_hi=%e  iter=%d/%d\n", x_lo, x_hi, iter, max_iter);
    }
    while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free (s);

    return x;
}

int find_moving_bracket_Bracket(double xguess_lo, double xguess_hi, double *root, pSIntegral *p)
{
    int has_root, iter, max_iter;
    double scale, dx, f_lo, f_hi;

    max_iter = pprec.si_findMovBracket_maxiter;
    scale = pprec.si_findMovBracket_scale;

    // the window moves from lo to hi
    dx = xguess_hi - xguess_lo;

    iter = 0;
    has_root = _FALSE_;
    do
    {
        f_lo = phi_SingleIntegral(xguess_lo, p);
        f_hi = phi_SingleIntegral(xguess_hi, p);

        // HVR_DEBUG
        //~ printf("x_lo=%g  x_hi=%g  f_lo=%g  f_hi=%g  dx=%g\n", xguess_lo, xguess_hi, f_lo, f_hi, dx);

        if( SIGN(f_lo) != SIGN(f_hi) )
            has_root = _TRUE_;
        else
        {
            dx *= scale;
            xguess_lo = xguess_hi;
            xguess_hi += dx;
        }

        iter++;
    }
    while( (iter<max_iter) && (has_root==_FALSE_) );

    if(has_root == _TRUE_)
        *root = find_bracket_Bracket(xguess_lo, xguess_hi, p);
    else
        PWARNING("moving bracket failed to find the extremum")

    return has_root;
}

int find_Brackets(pSIntegral *p)
{
    int i, n_beta, n_buffer;
    double x, dt1, dt2, x01, x02;
    double *beta_zeroes;
    CritPoint *cp, *cp2;

    n_beta = 0;
    n_buffer = 50;
    beta_zeroes = (double *)malloc(n_buffer*sizeof(double));

    // sort crit points
    //~ sort_x_CritPoint(p->n_points, p->points);

    // find leftmost bracket
    cp = p->points;
    if(p->t > cp->t)
    {
        // HVR_DEBUG
        //~ printf("  ** t=%g   cp->t=%g\n", p->t, cp->t);

        x = -MAX(sqrt(2*p->tau)+p->y, 2*ABS(cp->x1));
        x = find_root_Bracket(x, p);
        if(x > cp->x1)
            find_moving_bracket_Bracket(cp->x1, -ABS(x)+cp->x1, &x, p);
        beta_zeroes[n_beta++] = ABS(x);
    }

    // find rightmost bracket
    cp = p->points + p->n_points-1;
    if(p->t > cp->t)
    {
        // HVR_DEBUG
        //~ printf("  ** t=%g   cp->t=%g\n", p->t, cp->t);

        x = MAX(sqrt(2*p->tau)+p->y, 2*ABS(cp->x1));
        x = find_root_Bracket(x, p);
        if(x < cp->x1)
            find_moving_bracket_Bracket(cp->x1, ABS(x)+cp->x1, &x, p);
        beta_zeroes[n_beta++] = ABS(x);
    }

    // look for the other points
    for(i=1;i<p->n_points;i++)
    {
        cp  = p->points + i -1;
        cp2 = p->points + i;

        dt1 = cp->t - p->t;
        dt2 = cp2->t - p->t;

        if( SIGN(dt1) != SIGN(dt2) )
        {
            x01 = cp->x1;
            x02 = cp2->x1;

            x = find_bracket_Bracket(x01, x02, p);
            beta_zeroes[n_beta++] = ABS(x);

            // buffer full
            if(n_beta == n_buffer)
            {
                n_buffer *= 2;
                beta_zeroes = realloc(beta_zeroes, n_buffer*sizeof(double));
            }
        }
    }

    // sort zeroes (in growing radius)
    qsort(beta_zeroes, n_beta, sizeof(double), compare_double);

    // fill the brackets
    if(n_beta%2 != 0)
        PERROR("problem with the number of brackets, n_beta%%2 != 0 (n_beta=%d, tau=%g)", n_beta, p->tau)

    p->n_brackets = n_beta/2;
    p->brackets = (Bracket *)malloc(p->n_brackets*sizeof(Bracket));

    for(i=0;i<p->n_brackets;i++)
    {
        x01 = beta_zeroes[2*i];
        x02 = beta_zeroes[2*i+1];

        // check the brackets
        if( (dbeta_SingleIntegral(x01, p) < 0) || (dbeta_SingleIntegral(x02, p) > 0) )
            PWARNING("could not ensure proper left/right assignment of brackets (a=%g, b=%g, tau=%g)", x01, x02, p->tau)

        p->brackets[i].a = x01;
        p->brackets[i].b = x02;
    }

    if(p->n_brackets == 0)
        PERROR("no brackets found (tau=%g)", p->tau)

    free(beta_zeroes);

    return 0;
}

void free_Brackets(pSIntegral *p)
{
    free(p->brackets);
}


// ======  functions
// =================================================================

double phi_SingleIntegral(double x1, void *pintegral)
{
    pSIntegral *p = (pSIntegral *)pintegral;

    return phiFermat(p->y, x1, 0, p->Psi) - p->t;
}

double dphi_SingleIntegral(double x1, void *pintegral)
{
    double phi_derivs[N_derivs];
    pSIntegral *p = (pSIntegral *)pintegral;

    phiFermat_1stDeriv(phi_derivs, p->y, x1, 0, p->Psi);

    return phi_derivs[i_dx1];
}

void phi_dphi_SingleIntegral(double x1, void *pintegral, double *y, double *dy)
{
    double phi_derivs[N_derivs];
    pSIntegral *p = (pSIntegral *)pintegral;

    phiFermat_1stDeriv(phi_derivs, p->y, x1, 0, p->Psi);

    *y  = phi_derivs[i_0] - p->t;
    *dy = phi_derivs[i_dx1];
}

double beta_SingleIntegral(double r, void *pintegral)
{
    double phi_p, phi_m;
    double tmp1, tmp2;

    pSIntegral *p = (pSIntegral *)pintegral;

    tmp1 = 0.5*(r*r + p->y*p->y) - p->Psi->psi(r, 0, p->Psi->pLens) - p->t;
    tmp2 = r*p->y;

    phi_m = tmp1+tmp2;
    phi_p = tmp1-tmp2;

    return -phi_p*phi_m;
}

double dbeta_SingleIntegral(double r, void *pintegral)
{
    double phi_p, phi_m;
    double dphi_p, dphi_m;
    double phi_derivs[N_derivs];
    pSIntegral *p = (pSIntegral *)pintegral;

    phiFermat_1stDeriv(phi_derivs, p->y, r, 0, p->Psi);
    phi_p = phi_derivs[i_0] - p->t;
    dphi_p = phi_derivs[i_dx1];

    phiFermat_1stDeriv(phi_derivs, p->y, -r, 0, p->Psi);
    phi_m = phi_derivs[i_0] - p->t;
    dphi_m = phi_derivs[i_dx1];

    return -dphi_p*phi_m+phi_p*dphi_m;
}

double alpha_SingleIntegral(double r, void *pintegral)
{
    double b = beta_SingleIntegral(r, pintegral);

    if(b > 0)
        return 2*r/sqrt(b+EPS_SOFT);
    else
        return 0;
}


// ======  integrate alpha with brackets
// =================================================================

int integrate_dir_SingleIntegral(double *It, pSIntegral *p)
{
    int i;
    int status, sub_int_limit;
    double rmin, rmax;
    double rtol, atol;
    double result, error;
    gsl_function F;
    gsl_integration_workspace *w;

    // precision parameters
    rtol = pprec.si_dirSingInt.epsrel;
    atol = pprec.si_dirSingInt.epsabs;
    sub_int_limit = pprec.si_dirSingInt.n;

    F.function = &alpha_SingleIntegral;
    F.params = p;

    *It = 0;
    for(i=0;i<p->n_brackets;i++)
    {
        rmin = p->brackets[i].a + EPS_SOFT;
        rmax = p->brackets[i].b - EPS_SOFT;

        w = gsl_integration_workspace_alloc(sub_int_limit);
        status = gsl_integration_qags(&F, rmin, rmax, atol, rtol, sub_int_limit, w, &result, &error);
        gsl_integration_workspace_free(w);

        *It += result;
    }

    return status;
}


// ======  integrate with brackets and change of variables
// =================================================================

double integrand_SingleIntegral(double xi, void *pintegral)
{
    int i;
    double xi2, I, rmin, rmax, rmid, Delta;
    pSIntegral *p = (pSIntegral *)pintegral;

    I = 0;
    xi2 = xi*xi;

    for(i=0;i<p->n_brackets;i++)
    {
        rmin = p->brackets[i].a;
        rmax = p->brackets[i].b;
        rmid = 0.5*(rmin + rmax);

        Delta = (rmax-rmid);
        I += Delta*alpha_SingleIntegral(rmax - Delta*xi2, pintegral);

        Delta = (rmid-rmin);
        I += Delta*alpha_SingleIntegral(rmin + Delta*xi2, pintegral);
    }

    return 2*xi*I;
}

int integrate_qng_SingleIntegral(double *It, pSIntegral *p)
{
    int status;
    size_t neval;
    double rtol, atol, xi_min, error;
    gsl_function F;

    // precision parameters
    rtol = pprec.si_qngSingInt_epsrel;
    atol = pprec.si_qngSingInt_epsabs;
    xi_min = pprec.si_qngSingInt_ximin;

    F.function = integrand_SingleIntegral;
    F.params = p;

    status = gsl_integration_qng(&F, xi_min, 1, atol, rtol, It, &error, &neval);

    return status;
}

int integrate_qag_SingleIntegral(double *It, pSIntegral *p, int key)
{
    int status, sub_int_limit;
    double rtol, atol, xi_min, error;
    gsl_function F;
    gsl_integration_workspace *w;

    // precision parameters
    rtol = pprec.si_qagSingInt.epsrel;
    atol = pprec.si_qagSingInt.epsabs;
    sub_int_limit = pprec.si_qagSingInt.n;
    xi_min = pprec.si_qagSingInt_ximin;

    F.function = integrand_SingleIntegral;
    F.params = p;

    w = gsl_integration_workspace_alloc(sub_int_limit);
    status = gsl_integration_qag(&F, xi_min, 1, atol, rtol, sub_int_limit, key, w, It, &error);
    gsl_integration_workspace_free(w);

    return status;
}


// ======  driver to access all the methods
// =================================================================

double driver_SingleIntegral(double tau, double y, double tmin,
                             int n_points, CritPoint *points,
                             pNamedLens *pNLens, int method)
{
    int i, status;
    double I, I0, tau_real, tau_min;
    pSIntegral p;
    Lens Psi = init_lens(pNLens);

    // built-in step function
    if(tau < 0)
        return 0;

    // if tau is too small, we interpolate between 0 and tau_min
    tau_real = tau;
    tau_min = pprec.si_drivContour_taumin_over_y2*y*y;
    if(tau < tau_min)
        tau = tau_min;

    // check that the critpoints are sorted in growing x
    if(check_sorting(n_points, points) == _FALSE_)
        PERROR("the critical points in SingleIntegral must be sorted in x1-order")

    // initialize parameters
    p.y = y;
    p.tau = tau;
    p.t = tau + tmin;
    p.n_points = n_points;
    p.points = points;
    p.Psi = &Psi;

    find_Brackets(&p);

    // HVR_DEBUG
    //~ for(i=0;i<p.n_brackets;i++)
        //~ display_Bracket(p.brackets+i);

    if(method == m_integral_qng)
    {
        status = integrate_qng_SingleIntegral(&I, &p);

        if(status == GSL_ETOL)
        {
            PWARNING("could not reach desired accuracy with qng, switching to qag15 (tau=%g)", tau)
            status = integrate_qag_SingleIntegral(&I, &p, m_integral_g15);
        }
    }
    else if( (method >= m_integral_g15) && (method <= m_integral_g61) )
        status = integrate_qag_SingleIntegral(&I, &p, keys_gauss[method]);
    else if(method == m_integral_dir)
        status = integrate_dir_SingleIntegral(&I, &p);
    else
        PERROR("method %d not recognized in SingleIntegral", method)

    free_Brackets(&p);

    // linear interpolation when tau is very small
    if(tau_real < tau_min)
    {
        i = find_i_tmin_CritPoint(n_points, points);
        I0 = M_2PI*sqrt(magnification(points[i].x1, 0, &Psi));
        I = I + (I0 - I)*(1 - tau_real/tau_min);
    }

    return I;
}


// ======  get contours
// =================================================================

Contours *init_Contours(int n_contours, int n_points)
{
    int i;
    Contours *cnt = (Contours *)malloc(sizeof(Contours));

    cnt->n_contours = n_contours;
    cnt->n_points = n_points;

    cnt->x1 = (double **)malloc(n_contours*sizeof(double *));
    cnt->x2 = (double **)malloc(n_contours*sizeof(double *));

    for(i=0;i<n_contours;i++)
    {
        cnt->x1[i] = (double *)malloc(n_points*sizeof(double));
        cnt->x2[i] = (double *)malloc(n_points*sizeof(double));
    }

    return cnt;
}

void free_Contours(Contours *cnt)
{
    int i;

    for(i=0;i<cnt->n_contours;i++)
    {
        free(cnt->x1[i]);
        free(cnt->x2[i]);
    }

    free(cnt->x1);
    free(cnt->x2);
    free(cnt);
}

int fill_contour_SingleIntegral(Contours *cnt, pSIntegral *p)
{
    int i, j, half_n_points, n_points;
    double rmax, rmin, dr, r;
    double sth, cth, psi;

    n_points = cnt->n_points;
    half_n_points = n_points/2 ;

    if(cnt->n_points%2 != 0)
        half_n_points++;

    for(i=0;i<cnt->n_contours;i++)
    {
        rmax = p->brackets[i].b;
        rmin = p->brackets[i].a;
        dr = (rmax-rmin)/(half_n_points-1);

        for(j=0;j<half_n_points;j++)
        {
            r = rmin + j*dr;
            psi = p->Psi->psi(r, 0, p->Psi->pLens);
            cth = -(p->t + psi - 0.5*r*r - 0.5*p->y*p->y)/r/p->y;

            // avoid rounding-off errors for the sine
            if(cth > 1)
                cth = 1;
            if(cth < -1)
                cth = -1;

            sth = sqrt(1 - cth*cth);

            cnt->x1[i][j] = r*cth;
            cnt->x2[i][j] = r*sth;

            cnt->x1[i][n_points - j - 1] = r*cth;
            cnt->x2[i][n_points - j - 1] = -r*sth;
        }
    }

    return 0;
}

Contours *driver_get_contour_SingleIntegral(double tau, int n_cpoints, double y, double tmin,
                                            int n_points, CritPoint *points,
                                            pNamedLens *pNLens)
{
    pSIntegral p;
    Contours *cnt;
    Lens Psi = init_lens(pNLens);

    // check that the critpoints are sorted in growing x
    if(check_sorting(n_points, points) == _FALSE_)
        PERROR("the critical points in SingleIntegral must be sorted in x1-order")

    p.y = y;
    p.tau = tau;
    p.t = tau + tmin;
    p.n_points = n_points;
    p.points = points;
    p.Psi = &Psi;

    find_Brackets(&p);

    cnt = init_Contours(p.n_brackets, n_cpoints);
    fill_contour_SingleIntegral(cnt, &p);

    free_Brackets(&p);

    return cnt;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_spline.h>

#include "common.h"
#include "ode_tools.h"

// ==========================================================================

// New control routine
// ------------------------------------------------------------------

// replicate gsl std_control and add the conditional functions
typedef struct
{
    pCondODE *p;
    gsl_odeiv2_control *std_c;
}
cond_control_state_t;


// routine here (important stuff below)
static int control_set_driver_null (void *vstate, const gsl_odeiv2_driver * d)
{
  /* Dummy set function for those control objects that do not
     need pointer to driver object. */
  (void) vstate;
  (void) d;

  return GSL_SUCCESS;
}

static void *cond_control_alloc(void)
{
    cond_control_state_t *s = (cond_control_state_t *)malloc(sizeof(cond_control_state_t));

    if(s == 0)
        GSL_ERROR_NULL("failed to allocate space for std_control_state", GSL_ENOMEM);

    return s;
}

static int cond_control_init(void *vstate, double eps_abs, double eps_rel, double a_y, double a_dydt)
{
    gsl_odeiv2_control *std_c = gsl_odeiv2_control_standard_new(eps_abs, eps_rel, a_y, a_dydt);
    cond_control_state_t *s = (cond_control_state_t *) vstate;

    s->std_c = std_c;

    return GSL_SUCCESS;
}

static int cond_control_errlevel(void *vstate, const double y, const double dydt,
                                const double h, const size_t ind, double *errlev)
{
    cond_control_state_t *s = (cond_control_state_t *) vstate;

    return gsl_odeiv2_control_errlevel(s->std_c, y, dydt, h, ind, errlev);
}

static void cond_control_free(void *vstate)
{
    cond_control_state_t *s = (cond_control_state_t *) vstate;

    gsl_odeiv2_control_free(s->std_c);

    free(s);
}

// ------------------------------------------------------------------

// important stuff here
static int cond_control_hadjust(void *vstate, size_t dim, unsigned int ord,
                               const double y[], const double yerr[], const double yp[],
                               double *h)
{
    char add_cond_met;
    int status;
    double condition;
    cond_control_state_t *s = (cond_control_state_t *) vstate;
    pCondODE *p = s->p;
    gsl_odeiv2_control *c = s->std_c;

    // use the standard hadjust
    status = c->type->hadjust(c->state, dim, ord, y, yerr, yp, h);

    // step will be accepted
    if( (status != GSL_ODEIV_HADJ_DEC) && (p != NULL))
    {
        // HVR_DEBUG
        //~ for(i=0;i<dim;i++)
            //~ printf("y[%d]=%e     ", i, y[i]);
        //~ printf("h=%e\n", *h);

        // detect change of sign in the bracketing condition
        condition = p->brack_cond(y, yp, p);

        if( p->cond_initialized == _FALSE_ )
        {
            p->cond_old = condition;
            p->cond_initialized = _TRUE_;
        }

        if( SIGN(condition) != SIGN(p->cond_old) )
        {

            if(ABS(condition) < p->tol_brack)
            {
                if(p->add_cond != NULL)
                    add_cond_met = p->add_cond(y, yp, p);
                else
                    add_cond_met = _TRUE_;

                if(add_cond_met == _TRUE_)
                {
                    // HVR_DEBUG
                    //~ printf("condition = %e\n", condition);

                    p->cond_met = _TRUE_;
                }
                else
                    p->cond_old = condition;
            }
            else
            {
                // reduce step to meet the condition
                *h = *h/p->n_reduce_h;
                status = GSL_ODEIV_HADJ_DEC;

                // HVR_DEBUG
                //~ printf("h_new = %e\n", *h);
            }
        }
    }

    return status;
}

static const gsl_odeiv2_control_type cond_control_type =
{
    "conditional",   /* name */
    &cond_control_alloc,
    &cond_control_init,
    &cond_control_hadjust,
    &cond_control_errlevel,
    &control_set_driver_null,
    &cond_control_free
};

const gsl_odeiv2_control_type *gsl_odeiv2_control_conditional = &cond_control_type;

// ------------------------------------------------------------------

// more important stuff here
gsl_odeiv2_control*
gsl_odeiv2_control_conditional_new(double eps_abs, double eps_rel, pCondODE *p)
{
    int status;
    gsl_odeiv2_control *c;
    cond_control_state_t *s;

    c = gsl_odeiv2_control_alloc(gsl_odeiv2_control_conditional);
    status = gsl_odeiv2_control_init(c, eps_abs, eps_rel, 1.0, 0.0);

    s = (cond_control_state_t *)c->state;

    // initialize internal structure
    s->p = p;

    if (status != GSL_SUCCESS)
    {
        gsl_odeiv2_control_free(c);
        GSL_ERROR_NULL ("error trying to initialize control", status);
    }

    return c;
}

// ==========================================================================

// define new driver to use cond_control (copy pasted from gsl)
static gsl_odeiv2_driver *
driver_alloc (const gsl_odeiv2_system * sys, const double hstart,
              const gsl_odeiv2_step_type * T)
{
  /* Allocates and initializes an ODE driver system. Step and evolve
     objects are allocated here, but control object is allocated in
     another function.
   */

  gsl_odeiv2_driver *state;

  if (sys == NULL)
    {
      GSL_ERROR_NULL ("gsl_odeiv2_system must be defined", GSL_EINVAL);
    }

  state = (gsl_odeiv2_driver *) calloc (1, sizeof (gsl_odeiv2_driver));

  if (state == NULL)
    {
      GSL_ERROR_NULL ("failed to allocate space for driver state",
                      GSL_ENOMEM);
    }

  {
    const size_t dim = sys->dimension;

    if (dim == 0)
      {
        gsl_odeiv2_driver_free(state);
        GSL_ERROR_NULL
          ("gsl_odeiv2_system dimension must be a positive integer",
           GSL_EINVAL);
      }

    state->sys = sys;

    state->s = gsl_odeiv2_step_alloc (T, dim);

    if (state->s == NULL)
      {
        gsl_odeiv2_driver_free(state);
        GSL_ERROR_NULL ("failed to allocate step object", GSL_ENOMEM);
      }

    state->e = gsl_odeiv2_evolve_alloc (dim);
  }

  if (state->e == NULL)
    {
      gsl_odeiv2_driver_free(state);
      GSL_ERROR_NULL ("failed to allocate evolve object", GSL_ENOMEM);
    }

  if (hstart > 0.0 || hstart < 0.0)
    {
      state->h = hstart;
    }
  else
    {
      gsl_odeiv2_driver_free(state);
      GSL_ERROR_NULL ("invalid hstart", GSL_EINVAL);
    }

  state->h = hstart;
  state->hmin = 0.0;
  state->hmax = GSL_DBL_MAX;
  state->nmax = 0;
  state->n = 0;
  state->c = NULL;

  return state;
}

gsl_odeiv2_driver *
gsl_odeiv2_driver_alloc_conditional_new(const gsl_odeiv2_system * sys,
                                        const gsl_odeiv2_step_type * T,
                                        const double hstart,
                                        const double epsabs, const double epsrel,
                                        pCondODE *p)
{
  /* Initializes an ODE driver system with control object of type y_new. */

  gsl_odeiv2_driver *state = driver_alloc (sys, hstart, T);

  if (state == NULL)
    {
      GSL_ERROR_NULL ("failed to allocate driver object", GSL_ENOMEM);
    }

  if (epsabs >= 0.0 && epsrel >= 0.0)
    {
      state->c = gsl_odeiv2_control_conditional_new (epsabs, epsrel, p);

      if (state->c == NULL)
        {
          gsl_odeiv2_driver_free (state);
          GSL_ERROR_NULL ("failed to allocate control object", GSL_ENOMEM);
        }
    }
  else
    {
      gsl_odeiv2_driver_free (state);
      GSL_ERROR_NULL ("epsabs and epsrel must be positive", GSL_EINVAL);
    }

  /* Distribute pointer to driver object */

  gsl_odeiv2_step_set_driver (state->s, state);
  gsl_odeiv2_evolve_set_driver (state->e, state);
  gsl_odeiv2_control_set_driver (state->c, state);

  return state;
}

// ==========================================================================

pCondODE *init_pCondODE(void)
{
    pCondODE *p = (pCondODE *)malloc(sizeof(pCondODE));

    p->cond_met = _FALSE_;
    p->n_reduce_h = 10.;

    // initialize cond_old first time it enters into the driver
    p->cond_initialized = _FALSE_;
    p->cond_old = 0.;

    p->tol_brack = 1e-5;
    p->tol_add = 10*p->tol_brack;

    p->brack_cond = NULL;
    p->add_cond = NULL;
    p->params = NULL;

    return p;
}

void free_pCondODE(pCondODE *p)
{
    free(p);
}

int check_pCond(void *pCond)
{
    pCondODE *pc = (pCondODE *)pCond;

    if(pc != NULL)
        if(pc->cond_met == _TRUE_)
            return ODE_COND_MET;

    return GSL_SUCCESS;
}

// ==========================================================================

SolODE *init_SolODE(int n_eqs, int n_buffer)
{
    int i;
    SolODE *sol = (SolODE *)malloc(sizeof(SolODE));

    sol->n_buffer = n_buffer;
    sol->n_allocated = n_buffer;
    sol->n_points = 0;
    sol->n_eqs = n_eqs;

    sol->t = (double *)malloc(n_buffer*sizeof(double));
    sol->y = (double **)malloc(n_eqs*sizeof(double *));
    for(i=0;i<sol->n_eqs;i++)
        sol->y[i] = (double *)malloc(n_buffer*sizeof(double));

    return sol;
}

void free_SolODE(SolODE *sol)
{
    int i;

    for(i=0;i<sol->n_eqs;i++)
        free(sol->y[i]);
    free(sol->y);
    free(sol->t);

    free(sol);
}

void realloc_SolODE(SolODE *sol)
{
    int i;

    sol->n_allocated += sol->n_buffer;

    sol->t = (double *)realloc(sol->t, sizeof(double)*sol->n_allocated);
    for(i=0;i<sol->n_eqs;i++)
        sol->y[i] = (double *)realloc(sol->y[i], sizeof(double)*sol->n_allocated);
}

int fill_SolODE(double t, double *y, SolODE *sol)
{
    int i, j;

    j = sol->n_points;
    if(j == sol->n_allocated)
        realloc_SolODE(sol);
    sol->n_points++;

    sol->t[j] = t;
    for(i=0;i<sol->n_eqs;i++)
        sol->y[i][j] = y[i];

    return 0;
}

SolODE *interpolate_SolODE(int n_points, SolODE *sol)
{
    int i, j;
    double t0, tf, dt;
    SolODE *new_sol;
    gsl_interp_accel *acc;
    gsl_spline *spline;

    // create a new SolODE to store the interpolation
    new_sol = init_SolODE(sol->n_eqs, n_points);
    new_sol->n_points = n_points;

    // create temporal grid to evaluate the solution
    t0 = sol->t[0];
    tf = sol->t[sol->n_points-1];
    dt = (tf-t0)/(n_points-1);

    for(i=0;i<n_points;i++)
        new_sol->t[i] = t0 + i*dt;

    // build interpolation functions
    acc = gsl_interp_accel_alloc();
    //~ spline = gsl_spline_alloc(gsl_interp_linear, sol->n_points);
    spline = gsl_spline_alloc(gsl_interp_cspline, sol->n_points);

    for(j=0;j<sol->n_eqs;j++)
    {
        gsl_spline_init(spline, sol->t, sol->y[j], sol->n_points);

        for(i=0;i<n_points;i++)
            new_sol->y[j][i] = gsl_spline_eval(spline, new_sol->t[i], acc);
    }

    gsl_spline_free(spline);
    gsl_interp_accel_free(acc);
    free_SolODE(sol);

    return new_sol;
}

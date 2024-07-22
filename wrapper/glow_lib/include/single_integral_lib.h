#ifndef SINTEGRAL_LIB_H
#define SINTEGRAL_LIB_H

#include <gsl/gsl_integration.h>

typedef struct {
    double a, b;
} Bracket;

typedef struct {
    double y, tau, t;
    int n_points;
    CritPoint *points;
    int n_brackets;
    Bracket *brackets;
    Lens *Psi;
} pSIntegral;

typedef struct {
    int n_contours;
    int n_points;
    double **x1;  // x1[0-n_contours] has n_points
    double **x2;
} Contours;

enum methods_integral {m_integral_g15, m_integral_g21, m_integral_g31,
                       m_integral_g41, m_integral_g51, m_integral_g61,
                       m_integral_dir, m_integral_qng};

// =================================================================

int compare_double(const void *a, const void *b);
int check_sorting(int n_points, CritPoint *points);

// ======  Operate with brackets
// =================================================================
void display_Bracket(Bracket *br);
double find_root_Bracket(double x_guess, pSIntegral *p);
double find_bracket_Bracket(double a, double b, pSIntegral *p);
int find_moving_bracket_Bracket(double xguess_lo, double xguess_hi, double *root, pSIntegral *p);
int find_Brackets(pSIntegral *p);
void free_Brackets(pSIntegral *p);

// ======  Functions
// =================================================================
double phi_SingleIntegral(double x1, void *pintegral);
double dphi_SingleIntegral(double x1, void *pintegral);
void phi_dphi_SingleIntegral(double x1, void *pintegral, double *y, double *dy);
double beta_SingleIntegral(double r, void *pintegral);
double dbeta_SingleIntegral(double r, void *pintegral);
double alpha_SingleIntegral(double r, void *pintegral);

// ======  Integrate alpha with brackets
// =================================================================
int integrate_dir_SingleIntegral(double *It, pSIntegral *p);

// ======  Integrate with brackets and change of variables
// =================================================================
double integrand_SingleIntegral(double xi, void *pintegral);
int integrate_qng_SingleIntegral(double *It, pSIntegral *p);
int integrate_qag_SingleIntegral(double *It, pSIntegral *p, int key);

// ======  Driver to access all the methods
// =================================================================
double driver_SingleIntegral(double tau, double y, double tmin, int n_points, CritPoint *points,
                             pNamedLens *pNLens, int method);

// ======  Get contours
// =================================================================
Contours *init_Contours(int n_contours, int n_points);
void free_Contours(Contours *cnt);
int fill_contour_SingleIntegral(Contours *cnt, pSIntegral *p);
Contours *driver_get_contour_SingleIntegral(double tau, int n_cpoints, double y, double tmin,
                                            int n_points, CritPoint *points,
                                            pNamedLens *pNLens);


// =================================================================

#endif  // SINTEGRAL_LIB_H

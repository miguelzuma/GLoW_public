#ifndef LENSES_LIB_H
#define LENSES_LIB_H

typedef struct {
    double (*psi)(double x1, double x2, void *pLens);
    int (*psi_1stDerivs)(double *psi_derivs, double x1, double x2, void *pLens);
    int (*psi_2ndDerivs)(double *psi_derivs, double x1, double x2, void *pLens);
    void *pLens;
} Lens;

typedef struct {
    int lens_type;
    void *pLens;
} pNamedLens;

enum indices_spatial {i_x1, i_x2, N_dims};
enum indices_derivs {i_0, i_dx1, i_dx2, i_dx1dx1, i_dx2dx2, i_dx1dx2, N_derivs};

enum indices_lenses {i_SIS, i_CIS, i_PointLens, i_Ball, i_NFW, i_tSIS,
                     i_offcenterSIS, i_offcenterCIS, i_offcenterPointLens,
                     i_offcenterBall, i_offcenterNFW,
                     i_CombinedLens, i_Grid1d, i_SIE, i_Ext,
                     N_lenses};

extern char *names_lenses[];
extern Lens (*init_func_lenses[])(void *);
extern void (*free_func_lenses[])(pNamedLens *);

// =================================================================


// =================================================================
int x1x2_def(double *x_vec, double R, double alpha, double *x0_vec);
double phiFermat(double y, double x1, double x2, Lens *Psi);
int phiFermat_1stDeriv(double *phi_derivs, double y, double x1, double x2, Lens *Psi);
int phiFermat_2ndDeriv(double *phi_derivs, double y, double x1, double x2, Lens *Psi);
double magnification(double x1, double x2, Lens *Psi);

void rotate_vector(double *x_vec, double cos_th, double sin_th);
void rotate_gradient(double *f_derivs, double cos_th, double sin_th);
void rotate_gradient_hessian(double *f_derivs, double cos_th, double sin_th);

// check for singularities and cusps
double *add_cusp_sing(int *n, double *xvec, double x1, double x2);
double *get_cusp_sing(int *n, pNamedLens *pNLens);

double call_psi(double x1, double x2, Lens *Psi);
int call_psi_1stDerivs(double *psi_derivs, double x1, double x2, Lens *Psi);
int call_psi_2ndDerivs(double *psi_derivs, double x1, double x2, Lens *Psi);
// =================================================================


// =================================================================
Lens init_lens(pNamedLens *pNLens);
void free_pLens(pNamedLens *pNLens);
// =================================================================


// =================================================================
typedef struct
{
    double psi0;
} pLens_SIS;

pNamedLens* create_pLens_SIS(double psi0);
void free_pLens_SIS(pNamedLens* pNLens);
Lens init_lens_SIS(void *pLens);
double psi_SIS(double x1, double x2, void *pLens);
int psi_1stDerivs_SIS(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_SIS(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double xc1;
    double xc2;
    pLens_SIS pSIS;
} pLens_offcenterSIS;

pNamedLens* create_pLens_offcenterSIS(double psi0, double xc1, double xc2);
void free_pLens_offcenterSIS(pNamedLens* pNLens);
Lens init_lens_offcenterSIS(void *pLens);
double psi_offcenterSIS(double x1, double x2, void *pLens);
int psi_1stDerivs_offcenterSIS(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_offcenterSIS(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double psi0;
    double rc;
} pLens_CIS;

pNamedLens* create_pLens_CIS(double psi0, double rc);
void free_pLens_CIS(pNamedLens* pNLens);
Lens init_lens_CIS(void *pLens);
double psi_CIS(double x1, double x2, void *pLens);
int psi_1stDerivs_CIS(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_CIS(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double xc1;
    double xc2;
    pLens_CIS pCIS;
} pLens_offcenterCIS;

pNamedLens* create_pLens_offcenterCIS(double psi0, double rc, double xc1, double xc2);
void free_pLens_offcenterCIS(pNamedLens* pNLens);
Lens init_lens_offcenterCIS(void *pLens);
double psi_offcenterCIS(double x1, double x2, void *pLens);
int psi_1stDerivs_offcenterCIS(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_offcenterCIS(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    int n_sublenses_added;
    int n_sublenses;
    pNamedLens **psublenses;
    Lens *sublenses;
} pLens_CombinedLens;

int add_lens_CombinedLens(pNamedLens* new_pNLens, pNamedLens* combined_pNLens);
pNamedLens* create_pLens_CombinedLens(int n_sublenses);
void free_pLens_CombinedLens(pNamedLens* pNLens);
Lens init_lens_CombinedLens(void *pLens);
double psi_CombinedLens(double x1, double x2, void *pLens);
int psi_1stDerivs_CombinedLens(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_CombinedLens(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double psi0;
    double xs;
} pLens_NFW;

double F_NFW(double u);
pNamedLens* create_pLens_NFW(double psi0, double xs);
void free_pLens_NFW(pNamedLens* pNLens);
Lens init_lens_NFW(void *pLens);
double psi_NFW(double x1, double x2, void *pLens);
int psi_1stDerivs_NFW(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_NFW(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double xc1;
    double xc2;
    pLens_NFW pNFW;
} pLens_offcenterNFW;

pNamedLens* create_pLens_offcenterNFW(double psi0, double xs, double xc1, double xc2);
void free_pLens_offcenterNFW(pNamedLens* pNLens);
Lens init_lens_offcenterNFW(void *pLens);
double psi_offcenterNFW(double x1, double x2, void *pLens);
int psi_1stDerivs_offcenterNFW(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_offcenterNFW(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double psi0;
    double xc;
} pLens_PointLens;

pNamedLens* create_pLens_PointLens(double psi0, double xc);
void free_pLens_PointLens(pNamedLens* pNLens);
Lens init_lens_PointLens(void *pLens);
double psi_PointLens(double x1, double x2, void *pLens);
int psi_1stDerivs_PointLens(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_PointLens(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double xc1;
    double xc2;
    pLens_PointLens pPointLens;
} pLens_offcenterPointLens;

pNamedLens* create_pLens_offcenterPointLens(double psi0, double xc, double xc1, double xc2);
void free_pLens_offcenterPointLens(pNamedLens* pNLens);
Lens init_lens_offcenterPointLens(void *pLens);
double psi_offcenterPointLens(double x1, double x2, void *pLens);
int psi_1stDerivs_offcenterPointLens(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_offcenterPointLens(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double psi0;
    double b;
} pLens_Ball;

pNamedLens* create_pLens_Ball(double psi0, double b);
void free_pLens_Ball(pNamedLens* pNLens);
Lens init_lens_Ball(void *pLens);
double psi_Ball(double x1, double x2, void *pLens);
int psi_1stDerivs_Ball(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_Ball(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double xc1;
    double xc2;
    pLens_Ball pBall;
} pLens_offcenterBall;

pNamedLens* create_pLens_offcenterBall(double psi0, double b, double xc1, double xc2);
void free_pLens_offcenterBall(pNamedLens* pNLens);
Lens init_lens_offcenterBall(void *pLens);
double psi_offcenterBall(double x1, double x2, void *pLens);
int psi_1stDerivs_offcenterBall(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_offcenterBall(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double psi0;
    double xb;
} pLens_tSIS;

double F_tSIS(double u);
pNamedLens* create_pLens_tSIS(double psi0, double xb);
void free_pLens_tSIS(pNamedLens* pNLens);
Lens init_lens_tSIS(void *pLens);
double psi_tSIS(double x1, double x2, void *pLens);
int psi_1stDerivs_tSIS(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_tSIS(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    int n_grid;
    double *x_grid;
    double *y_grid;
    void *spline;
    void *acc;
} Interp1d;

typedef struct
{
    Interp1d *psi_interp;
    Interp1d *dpsi_interp;
    Interp1d *ddpsi_interp;
} pLens_Grid1d;

Interp1d* init_Interp1d(char *fname, int n_grid);
void free_Interp1d(Interp1d *f);
double eval_Interp1d(double x, Interp1d *f);
void load_Grid1d(char *fname, double *xvals, double *yvals, int nvals);

pNamedLens* create_pLens_Grid1d(char *fname, int n_grid);
void free_pLens_Grid1d(pNamedLens* pNLens);
Lens init_lens_Grid1d(void *pLens);
double psi_Grid1d(double x1, double x2, void *pLens);
int psi_1stDerivs_Grid1d(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_Grid1d(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double psi0;
    double q;
    double alpha;
    double ca;
    double sa;
    double xc1;
    double xc2;
} pLens_SIE;

pNamedLens* create_pLens_SIE(double psi0, double q, double alpha, double xc1, double xc2);
void free_pLens_SIE(pNamedLens* pNLens);
Lens init_lens_SIE(void *pLens);
double psi_a0_SIE(double x1, double x2, void *pLens);
int psi_1stDerivs_a0_SIE(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_a0_SIE(double *psi_derivs, double x1, double x2, void *pLens);

double psi_SIE(double x1, double x2, void *pLens);
int psi_1stDerivs_SIE(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_SIE(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
typedef struct
{
    double kappa;
    double gamma1;
    double gamma2;
} pLens_Ext;

pNamedLens* create_pLens_Ext(double kappa, double gamma1, double gamma2);
void free_pLens_Ext(pNamedLens* pNLens);
Lens init_lens_Ext(void *pLens);
double psi_Ext(double x1, double x2, void *pLens);
int psi_1stDerivs_Ext(double *psi_derivs, double x1, double x2, void *pLens);
int psi_2ndDerivs_Ext(double *psi_derivs, double x1, double x2, void *pLens);
// =================================================================


// =================================================================
//           New lens here
// =================================================================


// =================================================================
#endif  // LENSES_LIB_H

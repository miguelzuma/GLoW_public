Implementing a new lens
=======================

The best way to implement a new lens is to copy an existing implementation (e.g. CIS)
and make the appropiate changes. Below we outline the procedure and all the places
where changes should be made for a standard lens implementation.

We will assume that the new lens is called ``MyLens`` with parameters ``psi0``, ``p1``, 
``p2`` ...  (it is highly recommended to always add a parameter ``psi0`` that allows to 
scale the lensing potential if needed, even if it is one with the preferred choice of units).

.. ################################################################################################
.. ################################################################################################

Python
------

#. *Define the lens and its parameters (Mandatory)*. 

    .. code-block:: python
    
        # lenses.py
        class Psi_MyLens(PsiGeneral):
            """Lens object for MyLens.
            
            Parameters
            ----------
            p_phys : dict
                Physical parameters, with keys:
                
                * ``psi0`` (*float*) -- Normalization of the lens.
                * ``p1`` (*??*) -- ??
                * ``p2`` (*??*) -- ??
                * ...
            """
            def __init__(self, p_phys={}, p_prec={}):
                super().__init__(p_phys, p_prec)               
            
            def default_params(self):
                p_phys = {'name' : 'my lens',\
                          'psi0' : 1,\
                          'p1'   : 1.,\
                          'p2'   : 1.,\
                          ...}
                p_prec = {}
                return p_phys, p_prec
            
    All the parameters that we plan to use with the lens must be included in ``default_params()``. If we pass parameters that are not defined here the code will warn us.

#. *Check the input (Optional)*.

    .. code-block:: python
    
        # lenses.py
        class Psi_MyLens(PsiGeneral):
            ...
            def check_input(self):
                psi0 = self.p_phys['psi0']
                if psi0 < 0:
                    message = "psi0 = %g < 0 found" % psi0
                    raise LensException(message)

#. *Implement the lensing potential (Optional)*. If we are going to implement the lens in C, we can skip this step. It is however advisable to have an independent Python implementation. 

    .. code-block:: python
    
        # lenses.py
        class Psi_MyLens(PsiGeneral):
            def __init__(self, p_phys={}, p_prec={}):
                super().__init__(p_phys, p_prec)

            ...

            def psi(self, x1, x2):
                return ...
                
            def dpsi_vec(self, x1, x2):
                d1 = ...
                d2 = ...
                return d1, d2
                
            def ddpsi_vec(self, x1, x2):
                d11 = ...
                d12 = ...
                d22 = ...
                return d11, d12, d22

#. *Asymptotic information (Optional)*. If the derivative of the lens potential behaves asymptotically like a power-law,

    .. math::
        \psi'{\sim}\frac{A_\psi}{r^\gamma}, \quad r\to\infty
    
    there are two more quantities that we can define:
    
    .. math::
        \begin{align}
            \gamma_\text{asymp} &\equiv (\gamma+1)/2\ ,\\
            A_\text{asymp} &\equiv A_\psi/2^{\gamma_\text{asymp}}\ .
        \end{align}
                   
    These two quantities will help to regularize :math:`I(\tau)` and reduce the noise in the computation of :math:`F(w)`. If we choose the wrong parameters the results will still be consistent, but maybe more noisy than they should be. The regularization scheme is designed for power laws, but for lenses with logarithms, e.g. :math:`\psi'\sim r^{-\gamma}\log r`, it is still worth trying it since it will typically improve the results. We can include this information in the lens as 
    
    .. code-block:: python
    
        # lenses.py
        class Psi_MyLens(PsiGeneral):
            def __init__(self, p_phys={}, p_prec={}):
                ...        
                self.asymp_index     = ... # gamma_asymp
                self.asymp_amplitude = ... # A_asymp
                         
#. *Axisymmetric lenses (Optional)*. If our lens is axisymmetric we can simplify the calculation of the derivatives. First, we need to declare the lens as a subclass of :class:`~glow.lenses.PsiAxisym` and repeat all the steps above, changing only the implementation of the lensing potential:

    .. code-block:: python
    
        # lenses.py
        class Psi_MyLens(PsiAxisym):
            ...
            def psi_x(self, x):
                ...
                
            def dpsi_dx(self, x):
                ...
                
            def ddpsi_ddx(self, x):
                ...      
                
    The resulting lens will still have the ``psi(x1, x2)``, ``dpsi_vec(x1, x2)``, ``ddpsi_vec(x1, x2)`` methods but these will be computed from the radial derivatives defined above.

.. ################################################################################################
.. ################################################################################################


C and Cython wrapper
--------------------

#. *Declare the lens and data structures.* First go to ``wrapper/glow_lib/include/lenses.h`` and add a new identifier for our lens 

    .. code-block:: c
    
        // lenses.h
        enum indices_lenses {i_SIS, ...
                             ...
                             i_MyLens,
                             N_lenses};

    Define the structure that will store the parameters
    
    .. code-block:: c
    
        // lenses.h
        typedef struct
        {
            double psi0;
            double p1;
            double p2;
            ...
        } pLens_MyLens;
        
    and finally add the following definitions to ``lenses.h``.

    .. code-block:: c
    
        // lenses.h
        pNamedLens* create_pLens_MyLens(double psi0, double p1, double p2, ...);
        void free_pLens_MyLens(pNamedLens* pNLens);
        Lens init_lens_MyLens(void *pLens);
        double psi_MyLens(double x1, double x2, void *pLens);
        int psi_1stDerivs_MyLens(double *psi_derivs, double x1, double x2, 
                                 void *pLens);
        int psi_2ndDerivs_MyLens(double *psi_derivs, double x1, double x2, 
                                 void *pLens);
    
#. *Implement the lensing potential.* Now we must actually define these functions in ``wrapper/glow_lib/source/lenses.c``. The first step is to add a name for our lens *in the same position as our identifier* ``i_MyLens``
    
    .. code-block:: c
    
        // lenses.c
        char *names_lenses[] = {"SIS",
                                ...
                                "my lens"};

    Again *in the same position*, we must add the allocation and free function for our lens

    .. code-block:: c
    
        // lenses.c
        Lens (*init_func_lenses[N_lenses])(void *) = { init_lens_SIS, 
                                                       ...
                                                       init_lens_MyLens };
                                                       
        void (*free_func_lenses[N_lenses])(pNamedLens *) = { free_pLens_SIS, 
                                                             ...
                                                             free_pLens_MyLens };

    Next, we have to define the function that will initialize the parameters of the lens. If we don't need any special allocation, it will look like this
    
    .. code-block:: c
    
        // lenses.c                
        pNamedLens* create_pLens_MyLens(double psi0, double p1, double p2, ...)
        {
            pNamedLens *pNLens = (pNamedLens*)malloc(sizeof(pNamedLens));
            pLens_MyLens *pLens = (pLens_MyLens*)malloc(sizeof(pLens_MyLens));
            
            pLens->psi0 = psi0;
            pLens->p1 = p1;
            pLens->p2 = p2;
            ...
            
            pNLens->lens_type = i_MyLens;
            pNLens->pLens = pLens;
            
            return pNLens;
        }   
           
    If we didn't allocate anything else in the previous step, the next two functions don't need to be modified
    
    .. code-block:: c
    
        // lenses.c                
        void free_pLens_MyLens(pNamedLens *pNLens)
        {
            free(pNLens->pLens);
            free(pNLens);
        }
        
        Lens init_lens_MyLens(void *pLens)
        {
            Lens Psi;
            
            Psi.psi = psi_MyLens;
            Psi.psi_1stDerivs = psi_1stDerivs_MyLens;
            Psi.psi_2ndDerivs = psi_2ndDerivs_MyLens;
            Psi.pLens = pLens;
            
            return Psi;
        }
        
    Finally we have to define the lensing potential, with its first and second derivatives. The function ``psi`` will compute the lensing potential, ``psi_1stDerivs`` both the lensing potential and the first derivatives and ``psi_2ndDerivs`` all the preceding plus the second derivatives. Unfortunately the code must be copied and pasted.

    .. code-block:: c
    
        // lenses.c
        double psi_MyLens(double x1, double x2, void *pLens)
        {
            double psi;
            ...
            pLens_MyLens *p = (pLens_MyLens *)pLens;
            
            psi = p->psi0*...
            
            return psi;
        }
        
        int psi_1stDerivs_MyLens(double *psi_derivs, double x1, double x2, 
                                 void *pLens)
        {
            double psi, d1, d2;
            ...
            pLens_MyLens *p = (pLens_MyLens *)pLens;
            
            psi = p->psi0*...  
            d1 = ...           // dpsi/dx1
            d2 = ...           // dpsi/dx2
            
            psi_derivs[i_0] = psi;
            psi_derivs[i_dx1] = d1;
            psi_derivs[i_dx2] = d2;
            
            return 0;
        }
        
        int psi_2ndDerivs_MyLens(double *psi_derivs, double x1, double x2, 
                                 void *pLens)
        {
            double psi, d1, d2, d11, d22, d12;
            ...      
            pLens_MyLens *p = (pLens_MyLens *)pLens;
            
            psi = p->psi0*...
            d1 = ...
            d2 = ...
            d11 = ...      // ddpsi/dx1dx1
            d22 = ...      // ddpsi/dx2dx2
            d12 = ...      // ddpsi/dx1dx2
            
            psi_derivs[i_0] = psi;
            psi_derivs[i_dx1] = d1;
            psi_derivs[i_dx2] = d2;
            psi_derivs[i_dx1dx1] = d11;
            psi_derivs[i_dx2dx2] = d22;
            psi_derivs[i_dx1dx2] = d12;
            
            return 0;
        }

#. *Check for cusps/singularities (Optional)*. If the lens that we are implementing has any known problematic points (singularities or cusps) we can add this information to improve the computation. For axisymmetric lenses we can skip this step, in those cases the origin (and only the origin) will always be checked.

    .. code-block:: c
    
        // lenses.c
        double *get_cusp_sing(int *n, pNamedLens *pNLens)
        {
            ...
            if(lens_type == i_offcenterSIS)
            {
                x1 = ((pLens_offcenterSIS *)(pNLens->pLens))->xc1;
                x2 = ((pLens_offcenterSIS *)(pNLens->pLens))->xc2;        
                xvec = add_cusp_sing(n, xvec, x1, x2);
            }
            else if(lens_type == i_MyLens)
            {
                x1 = ((pLens_MyLens *)(pNLens->pLens))->xc1;
                x2 = ((pLens_MyLens *)(pNLens->pLens))->xc2;
                xvec = add_cusp_sing(n, xvec, x1, x2);
            }
            ...
        }
        
                        
#. *Create the wrapper.* The lens is ready to be used in C, but to be able to call it from Python we must create a wrapper. First, we go to ``wrapper/src/clenses.pxd`` and add the declaration of one of our previously defined C functions 

    .. code-block:: cython
    
        # clenses.pxd
        cdef extern from "lenses_lib.h":
            ...
            pNamedLens* create_pLens_MyLens(double psi0, double p1, double p2, ...)
            ...
            
    Next, in ``clenses.pyx`` we must define a function that translates the parameters stored in Python to the ones that we will use in C
    
    .. code-block:: cython
    
        # clenses.pyx
        cdef pNamedLens* convert_pphys_to_pLens_MyLens(p_phys):
            cdef double psi0 = p_phys['psi0']
            cdef double p1 = p_phys['p1']
            cdef double p2 = p_phys['p2']
            ...
            
            return clenses.create_pLens_MyLens(psi0, p1, p2, ...)
            
    Add the name of the new lens to the list of implemented lenses *with the same name* stored in ``p_phys['name']`` in the Python implementation.
    
    .. code-block:: cython
    
        # clenses.pyx
        implemented_lenses = ['SIS',\
                              ...
                              'my lens']

    Finally, add an entry in
    
    .. code-block:: cython
    
        # clenses.pyx
        cdef pNamedLens* convert_pphys_to_pLens(p_phys):
            name = p_phys['name']
            
            if name == 'SIS':
                return convert_pphys_to_pLens_SIS(p_phys)
            ...
            elif name == 'my lens':
                return convert_pphys_to_pLens_MyLens(p_phys)
            else:
                message = "WRAPPER ERROR: Unknown lens '%s'" % name
                raise ValueError(message)

.. ################################################################################################
.. ################################################################################################

Testing
-------
Manual testing
^^^^^^^^^^^^^^
There are two ways to manually test our implementation. First, to test 
directly our implementation in C, we first go to ``wrapper/glow_lib``. 
The Makefile in this folder will only compile the C code and running it as 
``make all`` will also compile some test files in the ``tests`` folder. We can 
then for instance modify and run ``tests/test_lenses`` to quickly check our 
implementation during the early development stage.

Once the lens is fully implemented, i.e. Python frontend + Cython wrapper, 
we can also test the C implementation directly from Python. We just need
to "decorate" the Python lens as

.. code-block:: python

    from glow import lenses
    from glow.wrapper import LensWrapper
    
    Psi_py = lenses.Psi_SIS()
    Psi_c  = LensWrapper(Psi_py)
    
and ``Psi_c`` will behave (almost) like a Python lens (it will lack some methods and arguments).

However, keep in mind that you may need to run ``make clean`` and then 
``make`` if you modify the C code and want to properly recompile the
wrapper (which takes some time, so the first method should be preferred during
early development).  

Automatic testing
^^^^^^^^^^^^^^^^^
There is a rudimentary script for automatic testing of lenses in 
``tests/test_lenses.py`` (in the main directory). At the bottom of the
file, new lenses can be added: 

.. code-block:: python

    # test_lenses.py
    ...
    if __name__ == '__main__':
        ...
        lenses = [lenses.Psi_SIS(),
                  ...,
                  lenses.Psi_MyLens()]
                  
This will compute the numerical derivatives of the lens potential in a grid
of points and compare it to the exact result, both for the Python and C 
implementations (if they are implemented, otherwise it will skip the test). It will
also compare both implementations and the internal consistency of the C code 
(i.e. we check that the copy-pasting in ``psi_1stDerivs`` and 
``psi_2ndDerivs`` didn't go wrong).

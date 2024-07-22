Implementing a new C module
===========================

The folder ``wrapper`` contains both C code under ``wrapper/glow_lib``
as well as a Cython interface in ``wrapper/src``. The structure is also 
very modular, but more fragmented than its Python counterpart, with different
modules dedicated to different tasks rather than to different steps in the pipeline.
Each module ``foo`` consists of the following files:

* ``wrapper/glow_lib/source/foo_lib.c``
* ``wrapper/glow_lib/include/foo_lib.h``
* ``wrapper/src/cfoo.pyx``
* ``wrapper/src/cfoo.pxd``

The (``*.c``, ``*.h``) contain the source code in C, the ``*.pyx`` 
are Cython source files and ``*.pxd`` are Cython header files, that contain
both definitions from the ``*.h`` C headers as well as headers of Cython 
functions to be used by other modules. Additionally, if we want to include a new module:

* We must add a target ``foo_lib.o`` in ``wrapper/glow_lib/Makefile``.
* We must add an entry to ``c_extensions`` in ``wrapper/setup.py``.
* To facilitate the access, every function to be used in the Python modules is imported in ``wrapper/__init__.py``.
* (Optional) We can also add a ``test_foo.c`` in the ``wrapper/glow_lib/tests`` folder (with the corresponding target in the ``Makefile``). This can prove especially useful during development for quick tests without recompiling the whole wrapper, but also afterwards, if problem arise during installation, it is convenient to have stand-alone tests of the C code.

The current list of modules is:

#. ``lenses``: mimicks the Python version. Contains the definitions of the lensing potential for the same lenses as the Python module, as well as functions common to all of them, like the Fermat potential and the magnification. 
#. ``roots``: contains all the algorithms related to root-finding and optimization, e.g. functions to find images or minimize the Fermat potential. 
#. ``single_contour``: implementation of the contour method for the case with a single critical point.
#. ``contour``: multicontour method for strong lensing with non-symmetric lenses.
#. ``single_integral``: implementation of the single integral method for axisymmetric lenses. 
#. ``area``: implementation of the binning/grid/area method. 
#. ``analytic_SIS``: analytic expressions for the SIS, both in time and frequency domain. 
#. ``special``: special functions not available in GSL, including at the moment the modified Struve function and the Fresnel integrals.
#. ``fourier``: definition of the regularizing functions, both in time and frequency domain, used in the computation of :math:`F(w)`.

Additional files:

* ``common.h`` and ``common.c``: macros common to all modules and the global structure containing the most of the precision parameters.
* ``ode_tools.c`` and ``ode_tools.h``: modified version of some GSL objects. In particular, a modified control system for the ODE integrator that is able to end the integration based on user-provided conditions, while accurately finding the stopping point. Applied in the code to the ``robust`` variation of the contour method, where we must detect that the contour has closed.

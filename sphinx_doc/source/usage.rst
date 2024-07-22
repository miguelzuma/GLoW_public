Usage
=====

.. _installation:

Installation
------------
Python
^^^^^^
The Python version of the code should work out of the box, provided all the
required packages are installed. The only non-standard requirement is ``mpmath``

.. code-block:: console

    $ pip install mpmath

which is only used to compute the analytic amplification factor for the point lens.

C and wrapper
^^^^^^^^^^^^^
The C version requires an additional step. Inside the main ``glow`` directory you must run

.. code-block:: console

    $ make

This will automatically compile the C code and the Cython wrapper. In this case the additional
requirements are Cython and the GNU Scientific Library (GSL). In many systems these can easily be
installed with

.. code-block:: console

    $ pip install cython
    $ sudo apt install libgsl-dev

OpenMP (OMP) support is also needed, but this should be already set up if you are using ``gcc``.

.. warning::

    If you are planning to use a Python environment, ``venv``, for this project, it must be
    activated *before* running any ``make`` command.

Other useful options are

.. code-block:: console

    $ make clean                  # clean-up
    $ cd wrapper/glow_lib; make   # compile only the C code

Documentation
^^^^^^^^^^^^^
This documentation relies on Sphinx, which can be installed with

.. code-block:: console

    $ pip install sphinx numpydoc sphinx_rtd_theme

The documentation can then be compiled with

.. code-block:: console

    $ make doc    # compile only the documentation
    $ make all    # compile documentation and code

This will create the HTML file ``docs/glow_doc.html``.

.. ####################################################################################################
.. ####################################################################################################

First steps
-----------
The main goal of the code is to compute the GW amplification factor for a lensing
event. Starting with the lensing potential :math:`\psi(\pmb{x})` for a given lens, the code
first computes an integral in the time domain (that also depends on the impact
parameter :math:`y`) that is then Fourier transformed to obtain the amplification factor
in the frequency domain.

.. math::
    \psi(\pmb{x})\qquad\to\qquad I(\tau)\qquad\to\qquad F(w)

This simple pipeline is mirrored in the code using three different modules with
three different base classes that represent each of these objects. Schematically, the
structure is

.. math::
    \begin{align}
        \psi(\pmb{x}) &\qquad\to\qquad \texttt{class Psi()}\\
        I(\tau)\,,\ y &\qquad\to\qquad \texttt{class It(Psi, y)}\\
        F(w) &\qquad\to\qquad \texttt{class Fw(It)}
    \end{align}

These objects are defined in three different Python modules:

* Computation of :math:`\psi(\pmb{x})`: ``lenses.py`` contains the definitions of :class:`~glow.lenses.PsiGeneral` for generic lenses and :class:`~glow.lenses.PsiAxisym` for axisymmetric lenses, i.e. :math:`\psi(\pmb{x})=\psi(x)`.
* Computation of :math:`I(\tau)`: ``time_domain.py`` and ``time_domain_c.py`` define :class:`~glow.time_domain.ItGeneral` and :class:`~glow.time_domain_c.ItGeneral_C`, respectively.
* Computation of :math:`F(w)`: ``freq_domain.py`` and ``freq_domain_c.py`` define ``FwGeneral()`` and ``FwGeneral_C()``, respectively.

Different lenses and algorithms are defined in each module as subclasses of these
base objects. While all modules rely on ``lenses.py``, ``time_domain.py``
and ``freq_domain.py`` are independent of their ``*_c.py`` counterparts.
The first ones are written fully in Python, while the latter rely on compiled code
(C and Cython) that is included in the ``wrapper`` folder. Although their content
do not exactly overlap, the ``*_c.py`` versions supersedes the Python-only
versions in almost every aspect, so its use is strongly recommended.

Let us now see how to use the code. First we must import the relevant modules,
then define each of the different objects, plugging them in the next one in the pipeline.
In the following example we will compute the amplification factor for a singular
isothermal sphere (SIS) for a lensing event with impact parameter :math:`y=1.2`.

.. code-block:: python

    from glow import lenses
    from glow import time_domain_c
    from glow import freq_domain_c

    # lensing potential for the singular isothermal sphere
    Psi = lenses.Psi_SIS()

    # time-domain integral with impact parameter y=1.2
    It = time_domain_c.It_SingleContour_C(Psi, y=1.2)

    # Fourier-transform It to obtain the amplification factor
    Fw = freq_domain_c.Fw_FFT_C(It)

After running this script, ``It`` and ``Fw`` will contain the values
of :math:`I(\tau)` and :math:`F(w)` evaluated in a grid of points, that can then be interpolated
and used to evaluate them at any point, like it is shown in the example below.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    ts = np.geomspace(1e-1, 1e2, 1000)
    plt.semilogx(ts, It(ts))
    plt.show()

    ## ------------------------------

    ws = np.geomspace(1e-2, 1e2, 1000)
    plt.semilogx(ws, Fw(ws))
    plt.show()

So far, we are not setting any parameter except the impact parameter :math:`y` (which is
mandatory). There are a lot of parameters that can be tuned for each object and all
of them are set through two dictionaries ``p_phys`` and ``p_prec``.
These dictionaries must be provided when we initialize the class, e.g.

.. code-block:: python

    p_phys = {'psi0' : 1.}
    p_prec = {}
    Psi = lenses.Psi_SIS(p_phys, p_prec)

Although there are a number of common parameters, in general the parameters vary from
one subclass to another in the module (i.e. for different lenses or for
different algorithms), as well as their default values. There are two handy ways to
know what are the available parameters and which ones is the object really using
to compute the results. First we can use the ``display_info()`` method, that
all the objects possess and will print in the screen these parameters. Secondly, we
can ``print`` the object. This will reproduce the exact Python call that we
need to replicate the object. In this case we will see the explicit
definition of the ``p_phys`` and ``p_prec`` dictionaries:

.. code-block:: python

    # internal parameters in pretty format
    Psi.display_info()
    It.display_info()
    Fw.display_info()

    # exact call needed to replicate Fw
    print('-'*50, '\n')
    print(Fw)

However, to know what these parameters actually do, it is better to check the internal documentation
of the function.

.. ####################################################################################################
.. ####################################################################################################

Tutorials
---------
.. toctree::
   tutorials/new_lens
   tutorials/new_module
   tutorials/debugging


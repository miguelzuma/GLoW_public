Frequency domain
================

Introduction
------------
(in preparation)

.. ###########################################################################################


.. _Regularization_theory:

Regularization
--------------
(in preparation)

See :class:`~glow.freq_domain.Fw_FFT_OldReg` for an implementation of the old regularization
scheme and :class:`~glow.freq_domain_c.Fw_FFT_C` for the new scheme.
The default parameters for the current build can be consulted 
:ref:`here <pyFw_FFT_default>` and :ref:`here <cFw_FFT_default>`.

.. ###########################################################################################


.. _AnalyticPointLens_theory:

Analytic point lens
-------------------
(in preparation)

See :class:`~glow.freq_domain.Fw_AnalyticPointLens` for an implementation using arbitrary-precision
software and :class:`~glow.freq_domain_c.Fw_AnalyticPointLens_C` for a much faster implementation
with reduced precision requirements. The default parameters for the current build can be consulted 
:ref:`here <pyFw_AnalyticPointLens_default>` and :ref:`here <cFw_AnalyticPointLens_default>`.

.. ###########################################################################################


.. _SemiAnalyticSIS_theory:

Semi-analytic SIS
-----------------
(in preparation)

See :class:`~glow.freq_domain.Fw_SemiAnalyticSIS` and :class:`~glow.freq_domain_c.Fw_SemiAnalyticSIS_C` for 
the implementation. The default parameters for the current build can be consulted 
:ref:`here <pyFw_SemiAnalyticSIS_default>` and :ref:`here <cFw_SemiAnalyticSIS_default>`.

.. ###########################################################################################


.. _DirectFT_theory:

Direct Fourier Transform
------------------------
(in preparation)

See :class:`~glow.freq_domain_c.Fw_DirectFT_C` for the implementation. The default parameters for the 
current build can be consulted :ref:`here <cFw_DirectFT_default>`.


.. ###########################################################################################


Module information
------------------
.. toctree::
    :maxdepth: 2
    
    py_base
    py_version
    c_base
    c_version
    default

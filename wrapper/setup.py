import numpy
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

# Note: if gcc complains (e.g -Wmaybe-uninitialized),
# add -Wno-maybe-uninitialized to extra_compile_args

clib_name    = "glow"
clib_lib     = "glow_lib/lib"
clib_include = "glow_lib/include"
extra_compile_args = ["-fopenmp", "-O3"]
extra_link_args    = ["-fopenmp"]

template_Extension = lambda name: Extension(name="c%s" % name, \
                                            sources=["src/c%s.pyx" % name], \
                                            libraries=[clib_name, "gsl", "gslcblas", "mvec", "m"], \
                                            library_dirs=[clib_lib], \
                                            include_dirs=[clib_include, numpy.get_include()], \
                                            extra_compile_args = extra_compile_args, \
                                            extra_link_args = extra_link_args, \
                                            depends=["src/c%s.pxd" % name], \
                                            define_macros=[("NPY_NO_DEPRECATED_API", 1)])

## ---------------------------------------------

c_extensions = []
c_extensions.append(template_Extension("lenses"))
c_extensions.append(template_Extension("single_contour"))
c_extensions.append(template_Extension("contour"))
c_extensions.append(template_Extension("analytic_SIS"))
c_extensions.append(template_Extension("roots"))
c_extensions.append(template_Extension("single_integral"))
c_extensions.append(template_Extension("area"))
c_extensions.append(template_Extension("special"))
c_extensions.append(template_Extension("fourier"))
c_extensions.append(template_Extension("common"))

setup(
    name="wrapper",
    ext_modules=cythonize(c_extensions, nthreads=4, annotate=False)
)

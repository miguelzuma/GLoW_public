#
# GLoW - setup.py
#
# Copyright (C) 2024, Hector Villarrubia-Rojo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import os, numpy, cython
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize

# Note: if gcc complains (e.g -Wmaybe-uninitialized),
# add -Wno-maybe-uninitialized to extra_compile_args

# make sure we use the same compiler as for the C lib
# note: only works for compiling! see below
os.environ["CC"] = "gcc"

# force the builder to use the same compiler for compiling and linking
class build_ext_with_compiler_detection(build_ext):
	def build_extensions(self):
		self.compiler.linker_so[0] = self.compiler.linker_exe[0]
		super().build_extensions()

## ---------------------------------------------------------------------

clib_name    = "glow"
clib_lib     = "glow_lib/lib"
clib_include = "glow_lib/include"
extra_compile_args = ["-fopenmp", "-O3"]
extra_link_args    = ["-fopenmp"]

template_Extension = lambda name: Extension(name="c%s" % name, \
                                            sources=["src/c%s.pyx" % name], \
                                            libraries=[clib_name, "gsl", "gslcblas", "m"], \
                                            library_dirs=[clib_lib], \
                                            include_dirs=[clib_include, numpy.get_include()], \
                                            extra_compile_args = extra_compile_args, \
                                            extra_link_args = extra_link_args, \
                                            depends=["src/c%s.pxd" % name], \
                                            define_macros=[("NPY_NO_DEPRECATED_API", 1)])

## ---------------------------------------------------------------------

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
    ext_modules=cythonize(c_extensions, nthreads=4, annotate=False),
    cmdclass = {'build_ext':build_ext_with_compiler_detection}
)

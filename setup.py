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

import os, sys

sys.path.append(os.path.dirname(__file__))
import configure

from setuptools import setup
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext

# otherwise the *.so are not stored
class BuildReorder(build):
    sub_commands = [('build_clib', build.has_c_libraries),
                    ('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_scripts', build.has_scripts),
                    ]

class BuildWrapper(build_ext):
    def run(self):
        configure.main(skip_parse=True)
        build_ext.run(self)

## ---------------------------------------------------------------------

setup(
    has_ext_modules = lambda: True,
    cmdclass = {
        "build": BuildReorder,
        "build_ext": BuildWrapper,
    },
)

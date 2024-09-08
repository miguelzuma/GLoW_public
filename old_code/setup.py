import subprocess
import os

import setuptools

from setuptools.command.build_ext import build_ext
from setuptools.command.build import build

class BuildReorder(build):
    sub_commands = [('build_clib', build.has_c_libraries),
                    ('build_ext', build.has_ext_modules),
                    ('build_py', build.has_pure_modules),
                    ('build_scripts', build.has_scripts),
                    ]

class BuildClibraries(build_ext):
    """
    Build procedure.

    The standard build procedure has been modified so as to run the Makefile for the C++ libraries
    that form the basis of the lens generation code.

    Methods
    -------
    run
        Executes Makefile and runs standard build procedure

    """

    def run(self):
        """
        Executes Makefile and runs standard build procedure. 
        The Makefile for the C libraries in located in the repository's `wrapper`
        folder. 

        """

        current_directory = os.getcwd()
        model_subfolder = f"{current_directory}/wrapper"

        os.chdir(model_subfolder)
        subprocess.run(['make', '-B'], check=True)
        os.chdir(current_directory)

        build_ext.run(self)

with open("README.md", "r", encoding="utf-8") as readme_contents:
    long_description = readme_contents.read()

with open("requirements.txt", "r", encoding="utf-8") as requirement_file:
    requirements = requirement_file.read().split("\n")

setuptools.setup(
    name = "Glow",
    version='1.1',
    author = "Hector Villarrubia Rojo",
    maintainer = "Hector Villarrubia Rojo",
    author_email = "herojo@aei.mpg.de",
    license = "GPL",
    description = ("Software package designed for generating fast and accurate lensed"
                   "gravitational wave signals given any lens distribution."),
    long_description = long_description,
    packages = [
        "glow",
        "glow.wrapper",
    ],
    package_dir={
        'glow': '.',
    },
    include_package_data=True,
    package_data = {
        'glow':  [
            'wrapper/*.so', 'sensitivities/*.txt']},
    has_ext_modules = lambda: True,
    install_requires = requirements,
    cmdclass = {
        "build": BuildReorder,
        "build_ext": BuildClibraries,
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "License :: OSI Approved :: GNU GPL License",
        ],
    python_requires = ">=3.9",
)

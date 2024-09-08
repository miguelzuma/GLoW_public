#
# GLoW - configure.py
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

import os, sys, re
import platform
import subprocess
import argparse
import shutil
import logging
from setuptools import sandbox
from contextlib import redirect_stdout, redirect_stderr

has_stdout_in_log = True
has_colours = True

root     = os.path.join(os.path.dirname(__file__), 'wrapper')
log_file = os.path.join(os.path.dirname(__file__), "configure.log")

logging.basicConfig(filename=log_file, filemode='a', level=logging.WARNING)
logger = logging.getLogger('logger')

########################################################################
# optional configuration options

if has_stdout_in_log is True:
    tmp_log_file = ".tmp_log"

    # create the tmp file that will store the temporal output
    tmp_file = os.path.join(root, tmp_log_file)
    with open(tmp_file, 'w', encoding="utf-8") as f:
        pass

    ## ************************************************************
    # hackish way to duplicate the output, to screen and to file
    old_print = print

    def _new_print(*args, sep=' ', end='\n'):
        old_print(*args, sep=sep, end=end, flush=True)

        args = _strip_color_args(args)
        with open(tmp_file, 'a', encoding="utf-8") as f:
            old_print(*args, sep=sep, end=end, file=f)

    print = _new_print
    ## ************************************************************
else:
    old_print = print

if has_colours is True:
    class bcolors:
        HEADER  = '\033[95m'
        OKBLUE  = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL    = '\033[91m'
        ENDC    = '\033[0m'

    def _strip_color_str(str_in):
        str_out = re.sub("\033\[(0|9[0-?])m", '', str_in)
        return str_out

    def _strip_color_args(args):
        args = list(args)
        for i, a in enumerate(args):
            args[i] = _strip_color_str(a)
        return tuple(args)
else:
    class bcolors:
        HEADER  = ''
        OKBLUE  = ''
        OKGREEN = ''
        WARNING = ''
        FAIL    = ''
        ENDC    = ''

    def _strip_color_args(args):
        return args

########################################################################

########################################################################
# class definitions

class SysInfo():
    def __init__(self):
        self.check_platform()
        self.check_conda_env()
        self.check_venv()

    ## ************************************
    def check_platform(self):
        self.os_short = platform.system()

        self.has_Linux = False
        self.has_Mac   = False
        self.has_Win   = False
        if self.os_short == 'Linux':
            self.has_Linux = True
        elif self.os_short == 'Darwin':
            self.has_Mac = True
        elif self.os_short == 'Windows':
            self.has_Win = True
        else:
            message = 'OS not recognized (%s)' % self.os_short
            raise RuntimeError(message)

    def check_conda_env(self):
        # check if we are in a conda environment
        try:
            self.conda_env_name = os.environ['CONDA_DEFAULT_ENV']
        except KeyError:
            self.conda_env_name = None

        try:
            self.conda_env_prefix = os.environ['CONDA_PREFIX']
        except KeyError:
            self.conda_env_prefix = None

        if self.conda_env_name is not None and self.conda_env_prefix is not None:
            self.has_conda_env = True
        else:
            self.has_conda_env = False

    def check_venv(self):

        # check if we are in a virtual environment
        try:
            self.venv_name = os.environ['VIRTUAL_ENV']
        except KeyError:
            self.venv_name = None

        if self.venv_name:
            self.has_venv = True
        else:
            self.has_venv = False
    ## ************************************

    def get_default_compiler(self):
        if self.has_Mac:
            compiler = "clang"
        else:
            compiler = "gcc"
        return compiler

    def get_omp_flags(self):
        cflags = [""]
        lflags = [""]

        if self.has_Linux:
            cflags = ["-fopenmp"]
            lflags = ["-fopenmp"]
        elif self.has_Mac:
            cflags = ["-Xclang", "-fopenmp"]
            lflags = ["-lomp"]
        else:
            pass

        return cflags, lflags

    def display(self):
        print(' - OS type:', self.os_short)

        print(' - OS version: ', end='')
        if self.has_Linux:
            info = platform.freedesktop_os_release()
            print(info['PRETTY_NAME'])
        elif self.has_Mac:
            info = platform.mac_ver()
            print('Darwin', info[0])
        elif self.has_Win:
            info = platform.win32_ver()
            print('Windows', info[0])
        else:
            message = 'OS not recognized (%s)' % self.os_short
            raise RuntimeError(message)

        print(' - Architecture:',  platform.architecture()[0])
        print(' - Full platform:', platform.platform())
        print(' - Processor:',     platform.processor())

        print('')
        print(' - Python version:', platform.python_version())
        print(' - Python implementation:', platform.python_implementation()+', '+platform.python_compiler())

        if self.has_conda_env:
            print('')
            print(' - Conda env name:', self.conda_env_name)
            print(' - Conda prefix:', self.conda_env_prefix)

        if self.has_venv:
            print('')
            print(' - Virtual env:', self.venv_name)

class Makefile_from_Args():
    def __init__(self, root_path, args, sys_info, has_cc=True, has_args=True, has_opt=True, has_omp=True, has_gsl=True):
        self.args = args
        self.sys_info = sys_info
        self.mk_file = os.path.join(root_path, "Makefile")
        self.mk_lines = self.create_bak()

        self.update_funcs = []
        if has_cc is True:
            self.update_funcs.append(self.update_cc)
        if has_args is True:
            self.update_funcs.append(self.update_args)
        if has_opt is True:
            self.update_funcs.append(self.update_opt)
        if has_omp is False:
            self.update_funcs.append(self.remove_omp)
        else:
            self.update_funcs.append(self.update_omp)
        if has_gsl is True:
            self.update_funcs.append(self.update_gsl)

        self.update_new_makefile()

    ## ****************************
    def create_bak(self):
        # copy from the standard template or create it if it doesn't exist
        try:
            f = open(self.mk_file+'.bak', 'r')
        except FileNotFoundError:
            shutil.copyfile(self.mk_file, self.mk_file+'.bak')
            f = open(self.mk_file+'.bak', 'r')

        mk_lines = f.readlines()
        f.close()

        return mk_lines

    def update_cc(self, head, flags):
        # update compiler
        if head == "CC":
            if self.args.compiler is not None:
                flags = [self.args.compiler]
        return flags

    def update_args(self, head, flags):
        # add optional flags
        for FLAGS, ARGS in zip(("CFLAGS", "LDFLAGS"), (self.args.CFLAGS, self.args.LDFLAGS)):
            if ARGS is not None:
                if head == FLAGS:
                    for f in ARGS:
                        flags.append('-'+f)
        return flags

    def update_opt(self, head, flags):
        # update optimization
        if head == "CFLAGS" and self.args.optimization is not None:
            for j, f in enumerate(flags):
                if f[:2] == '-O':
                    flags.pop(j)
            flags.append(f"-O{self.args.optimization}")
        return flags

    def remove_omp(self, head, flags):
        # remove parallel option if OMP not found
        for FLAGS in ("CFLAGS", "LDFLAGS"):
            if head == FLAGS:
                try:
                    flags.remove('-fopenmp')
                except ValueError:
                    pass
        return flags

    def update_omp(self, head, flags):
        if self.sys_info.has_Linux:
            pass
        elif self.sys_info.has_Mac:
            if (head == "CFLAGS") or (head == "LDFLAGS"):
                # check if we actually removed the flag,
                # i.e. if we even need the new one
                n = len(flags)
                flags = self.remove_omp(head, flags)
                if len(flags) != n:
                    cflags, lflags = self.sys_info.get_omp_flags()

                    if head == "CFLAGS":
                        for fl in cflags:
                            flags.append(fl)
                    if head == "LDFLAGS":
                        for fl in lflags:
                            flags.append(fl)
        return flags

    def update_gsl(self, head, flags):
        # add gsl directories
        if self.args.gsl_dir is not None:
            if head == 'CFLAGS':
                gsl_include = os.path.join(self.args.gsl_dir, "include")
                flags.insert(0, '-I%s' % gsl_include)

            if head == 'LDFLAGS':
                gsl_lib = os.path.join(self.args.gsl_dir, "lib")
                flags.insert(0, '-L%s' % gsl_lib)
                flags.append('-Wl,-rpath,%s' % gsl_lib)

        return flags
    ## ****************************

    def update_new_makefile(self):
        with open(self.mk_file, 'w', encoding="utf-8") as fp:
            for i, l in enumerate(self.mk_lines):
                new_line = l
                tmp_l = l.split('=', 1)

                if len(tmp_l) > 1:
                    head  = tmp_l[0].strip()
                    flags = (tmp_l[1:][0]).split()

                    for updt_func in self.update_funcs:
                        flags = updt_func(head, flags)

                    new_line = head + ' = ' + " ".join(flags) + '\n'

                    # edge case for valgrind with options
                    if head.split()[0] == 'valgrind':
                        new_line = l

                # remove parallel compilation flags
                new_line = re.sub('-j ?[1-9]? ?', '', new_line)

                fp.write(new_line)

class Setup_from_Args():
    def __init__(self, root_path, args, sys_info, has_cc=True, has_args=True, has_opt=True, has_omp=True, has_gsl=True):
        self.args = args
        self.sys_info = sys_info
        self.setup_file = os.path.join(root_path, "setup.py")
        self.setup_lines = self.create_bak()

        self.update_funcs = []
        if has_cc is True:
            self.update_funcs.append(self.update_cc)
        if has_args is True:
            self.update_funcs.append(self.update_args)
        if has_opt is True:
            self.update_funcs.append(self.update_opt)
        if has_omp is False:
            self.update_funcs.append(self.remove_omp)
        else:
            self.update_funcs.append(self.update_omp)
        if has_gsl is True:
            self.update_funcs.append(self.update_gsl)
        self.update_funcs.append(self.final_update)

        self.update_new_setup()

    ## ****************************
    def create_bak(self):
        # copy from the standard template or create it if it doesn't exist
        try:
            f = open(self.setup_file+'.bak', 'r')
        except FileNotFoundError:
            shutil.copyfile(self.setup_file, self.setup_file+'.bak')
            f = open(self.setup_file+'.bak', 'r')

        setup_lines = f.readlines()
        f.close()

        return setup_lines

    def update_cc(self, head, flags):
        # update compiler
        updated = False
        if head == 'os.environ["CC"]':
            if self.args.compiler is not None:
                flags = '"'+self.args.compiler+'"\n'
                updated = True
        return updated, flags

    def update_args(self, head, flags):
        # add optional flags
        updated = False
        if flags.strip()[0] == '[':
            # supress warnings from ld due to the outdated Mac versions in Cython's distutils
            if (head == "extra_link_args") and (self.sys_info.has_Mac is True):
                flags = self.add_flag('-Wl,-w', flags)
                updated = True

            for FLAGS, ARGS in zip(("extra_compile_args", "extra_link_args"), (self.args.CFLAGS, self.args.LDFLAGS)):
                if ARGS is not None:
                    if head == FLAGS:
                        for f in ARGS:
                            flags = self.add_flag('-'+f, flags)
                        updated = True
        return updated, flags

    def update_opt(self, head, flags):
        # update optimization
        updated = False
        if flags.strip()[0] == '[':
            if head == "extra_compile_args" and self.args.optimization is not None:
                flags = self.remove_flag_starting_by('-O', flags)
                flags = self.add_flag(f"-O{self.args.optimization}", flags)
                updated = True
        return updated, flags

    def remove_omp(self, head, flags):
        # remove parallel option if OMP not found
        updated = False
        if flags.strip()[0] == '[':
            for FLAGS in ("extra_compile_args", "extra_link_args"):
                if head == FLAGS:
                    flags = self.remove_flag("-fopenmp", flags)
                    updated = True
        return updated, flags

    def update_omp(self, head, flags):
        updated = False

        if self.sys_info.has_Linux:
            pass
        elif self.sys_info.has_Mac:
            if (head == "extra_compile_args") or (head == "extra_link_args"):
                updated, flags = self.remove_omp(head, flags)

                # check if we actually removed the flag,
                # i.e. if we even need the new one
                if updated:
                    cflags, lflags = self.sys_info.get_omp_flags()

                    if head == "extra_compile_args":
                        for fl in cflags:
                            flags = self.add_flag(fl, flags)
                    if head == "extra_link_args":
                        for fl in lflags:
                            flags = self.add_flag(fl, flags)

        return updated, flags

    def update_gsl(self, head, flags):
        # add gsl directories
        updated = False
        if flags.strip()[0] == '[':
            if self.args.gsl_dir is not None:
                if head == 'extra_compile_args':
                    gsl_include = os.path.join(self.args.gsl_dir, "include")
                    flags = self.add_flag('-I%s' % gsl_include, flags, index=0)
                    updated = True

                if head == 'extra_link_args':
                    gsl_lib = os.path.join(self.args.gsl_dir, "lib")
                    flags = self.add_flag('-L%s' % gsl_lib, flags, index=0)
                    flags = self.add_flag('-Wl,-rpath,%s' % gsl_lib, flags)
                    updated = True

        return updated, flags

    def final_update(self, head, flags):
        # check if the list of flags is empty and then change it by None
        updated = False
        if flags.strip()[0] == '[':
            if head == "extra_compile_args" or head == "extra_link_args":
                if flags[:4] == '[""]':
                    flags = 'None\n'
                    updated = True
                else:
                    # strip the list of "" elements
                    flags = re.sub('"", ', '', flags)
        return updated, flags

    def add_flag(self, name, flags, index=None):
        new_flags = eval(flags)
        if index is None:
            new_flags.append(name)
        else:
            new_flags.insert(index, name)

        return '["' + '", "'.join(new_flags) + '"]\n'

    def remove_flag(self, name, flags):
        new_flags = eval(flags)

        try:
            new_flags.remove(name)
        except ValueError:
            pass

        return '["' + '", "'.join(new_flags) + '"]\n'

    def remove_flag_starting_by(self, start_name, flags):
        new_flags = eval(flags)

        regex = re.compile(start_name)
        new_flags = [f for f in new_flags if not regex.match(f)]

        return '["' + '", "'.join(new_flags) + '"]\n'
    ## ****************************

    def update_new_setup(self):
        with open(self.setup_file, 'w', encoding="utf-8") as fp:
            for i, l in enumerate(self.setup_lines):
                new_line = l
                tmp_l = l.split('=', 1)

                updated = False
                if len(tmp_l) > 1:
                    head  = tmp_l[0].strip()
                    flags = tmp_l[1:][0]

                    for updt_func in self.update_funcs:
                        updt, flags = updt_func(head, flags)
                        updated = updated or updt

                    if updated is True:
                        new_line = head + ' = ' + flags

                # use only one thread for the extensions (more for dev mode)
                new_line = re.sub('nthreads=[1-9], ', '', new_line)

                fp.write(new_line)

########################################################################

########################################################################
# function definitions

def get_header(string, begin='\n', end='\n'):
    header = begin+"="*60+'\n'
    header += '   %s' % string + '\n'
    header += "="*60+end
    return header

def print_header(string, begin='\n', end='\n'):
    print(get_header(string, begin='\n', end='\n'))

def add_test(test_name, test_file, root, args, sys_info, log_file=log_file, runit=True, clean=True, has_omp=True):
    print('%s: ' % test_name, end='')

    success = True
    test_folder = os.path.join(root, test_file)
    test_exec   = os.path.join(test_folder, test_file)

    # update the makefile with the user options if needed
    if args is not None:
        Makefile_from_Args(test_folder, args, sys_info, has_omp=has_omp)

    if clean is True:
        proc = subprocess.run(("make -C %s clean" % test_folder).split(), capture_output=True)

    proc = subprocess.run(("make -C %s" % test_folder).split(), capture_output=True)
    returncode = proc.returncode
    stdout = proc.stdout.decode('utf-8')
    stderr = proc.stderr.decode('utf-8')

    if returncode == 0:
        status = "%sok%s" % (bcolors.OKGREEN, bcolors.ENDC)
    else:
        status = "%sfail%s" % (bcolors.FAIL, bcolors.ENDC)
        success = False

    if returncode == 0 and runit is True:
        proc = subprocess.run(test_exec, capture_output=True)
        returncode2 = proc.returncode
        stdout += proc.stdout.decode('utf-8')
        stderr += proc.stderr.decode('utf-8')

        status += ', '
        if returncode2 == 0:
            status += "%sok%s" % (bcolors.OKGREEN, bcolors.ENDC)
        else:
            status += "%sfail%s" % (bcolors.FAIL, bcolors.ENDC)
            success = False

    print(status)

    with open(log_file, 'a') as f:
        out = get_header(test_name)
        out += 'stdout:\n' + '-'*40 + '\n' + stdout + '\n'
        out += 'stderr:\n' + '-'*40 + '\n' + stderr
        out += '\n'
        f.write(out)

    return success

def get_full_log():
    if has_stdout_in_log is True:
        log_partial = log_file
        with open(log_partial, 'r') as f:
            log_lines = f.readlines()

        full_log = os.path.join(root, tmp_file)
        with open(full_log, 'a', encoding="utf-8") as f:
            f.write('#'*60+'\n'+'#'*60+'\n')
            for l in log_lines:
                f.write(l)

        os.replace(full_log, log_partial)

def get_parser(sys_info, skip_parse=False):
    # set up parser
    parser = argparse.ArgumentParser()
    default_gsl = sys_info.conda_env_prefix

    # default values if we are in an environment
    default_compiler = sys_info.get_default_compiler()

    # parser options
    parser.add_argument("-gsl", "--gsl_dir",
                        help="GSL installation directory",
                        default=default_gsl)
    parser.add_argument("-cc",  "--compiler",
                        help="Compiler to be used",
                        default=default_compiler)
    parser.add_argument("-O",   "--optimization",
                        help="Optimization level")
    parser.add_argument("-cf",  "--CFLAGS",
                        help="Additional compiling flags",
                        action='append')
    parser.add_argument("-lf",  "--LDFLAGS",
                        help="Additional linking flags",
                        action='append')

    if skip_parse == True:
        args = ""
    else:
        args = None

        if len(sys.argv) > 1:
            message  = "\nRunning configuration script with custom options:\n"
            message += " ~$ " + sys.executable + " " + " ".join(sys.argv) + "\n"

            with open(log_file, 'a', encoding="utf-8") as f:
                f.write(message)

    return parser.parse_args(args=args)

def build(skip_parse=False):
    # initialize the log file
    with open(log_file, 'w', encoding="utf-8") as f:
        pass

    #--------------------------------------------------
    print_header("PLATFORM INFO")

    sys_info = SysInfo()
    sys_info.display()

    # glow doesn't work in Windows for now
    if sys_info.has_Win:
        message = "automatic installation not available for Windows. Try following the instruction for manual installation."
        raise Exception(message)

    args = get_parser(sys_info, skip_parse)
    #--------------------------------------------------

    #--------------------------------------------------
    print_header("STARTING TESTS")

    root_tests = os.path.join(root, 'tests')
    has_gsl     = add_test(" 1) Testing GSL", "test_gsl", root_tests, args, sys_info)
    has_omp     = add_test(" 2) Testing OMP", "test_omp", root_tests, args, sys_info)
    has_complex = add_test(" 3) Testing complex", "test_complex", root_tests, args, sys_info)

    # abort if GSL is not working
    if has_gsl is False:
        advice = "GSL is not correctly configured, check that it is installed and linked appropiately."
        print('\n 1) %sCritical error%s. %s' % (bcolors.FAIL, bcolors.ENDC, advice))

    # proceed even if OMP is not working
    if has_omp is False:
        advice = "OMP is not correctly configured.. The installation will proceed without parallelization."
        print('\n 2) %sNon-critical error%s. %s'  % (bcolors.WARNING, bcolors.ENDC, advice))

    # abort if there is any problem with complex.h
    if has_complex is False:
        advice = "Some problem occurred with complex.h standard library."
        print('\n 3) %sCritical error%s. %s' % (bcolors.FAIL, bcolors.ENDC, advice))

    if has_gsl is False:
        message = "GSL library is not correctly configured. See configure.log for more details."
        raise Exception(message)

    if has_complex is False:
        message = "some proble occurred with complex.h. See configure.log for more details."
        raise Exception(message)
    #--------------------------------------------------

    #--------------------------------------------------
    print_header("COMPILING GLOW C LIBRARY")
    has_glow = add_test(" 4) Compiling GLoW lib", "glow_lib", root, args, sys_info, has_omp=has_omp, runit=False)

    if has_glow is False:
        message = "GLoW C library installation failed."
        print('\n 4) %sCritical error%s. %s' % (bcolors.FAIL, bcolors.ENDC, message))
        raise Exception(message)
    #--------------------------------------------------

    #--------------------------------------------------
    # create setup.py.bak and add the flags
    print_header("BUILDING WRAPPER")
    Setup_from_Args(root, args, sys_info, has_omp=has_omp)

    # change Makefile to remove -j flags
    Makefile_from_Args(root, args, sys_info)

    # add cython version
    try:
        import cython
        version = cython.__version__
    except ModuleNotFoundError:
        message = "Cython not found. Try installing it with 'pip install cython'."
        raise ModuleNotFoundError(message)

    print(" This step may take a couple of minutes.")
    print(" Check configure.log for details.\n")
    test_name = " 5) Cythonizing and compiling (v%s)" % version
    print("%s: " % test_name, end='')

    with open(log_file, 'a') as f:
        out = get_header(test_name)
        out += '\n'
        f.write(out)

        # setup.py must be run in sandbox mode (within python) for at least two reasons:
        # 1) if we manually invoke the Makefile, we might be using a different python version
        # 2) if numpy and cython are not installed, pip will try to install them and they will
        #    be available for this script if we add them to setup_requires, but they cannot
        #    be used by setup.py if we call it as an external script
        with redirect_stdout(f), redirect_stderr(f):
            cwd = os.getcwd()
            os.chdir(root)
            has_wrapper = sandbox.run_setup('setup.py', ['build_ext', '--inplace'])
            os.chdir(cwd)

    if has_wrapper is False:
        status = "%sfail%s" % (bcolors.FAIL, bcolors.ENDC)
        print(status)

        message = "wrapper installation failed."
        print('\n 5) %sCritical error%s. %s' % (bcolors.FAIL, bcolors.ENDC, message))
        raise Exception(message)
    else:
        status = "%sok%s" % (bcolors.OKGREEN, bcolors.ENDC)
        print(status)
    #--------------------------------------------------

def main(skip_parse=True):
    # add error to log and then combine
    try:
        build(skip_parse)
    except Exception as e:
        logging.exception("An exception was thrown!")
        raise e
    finally:
        print('')
        get_full_log()

########################################################################

if __name__ == '__main__':
    main(skip_parse=False)

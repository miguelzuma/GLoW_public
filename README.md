# GLoW: Gravitational Lensing of Waves

(To appear, cite 2409.xxxxx)

Used in
* [Savastano+ 23](https://inspirehep.net/literature/2667175)
* [Zumalacarregui 24](https://inspirehep.net/literature/2781293)

The online documentation can be found
[here](https://miguelzuma.github.io/glow/index.html).

## Installation

![GLoW-Light](./sphinx_doc/diagrams/diagram_simp.png#gh-light-mode-only)
![GLoW-Dark](./sphinx_doc/diagrams/diagram_simp_dark.png##gh-dark-mode-only)

The pure Python version of the code should work out of the box. It only requires standard scientific
packages like ``numpy`` and ``scipy``.

The C version requires an additional step. Inside the main ``glow`` directory you must run
```console
$ make
```
This will automatically compile the C library and the Cython wrapper. In this case the additional
requirements are Cython and the GNU Scientific Library (GSL). In many systems these can be easily
installed with
```console
$ pip install cython
$ sudo apt install libgsl-dev
```

> [!WARNING]
> If you are planning to use a Python environment, ``venv``, for this project, it must be activated *before* running any ``make`` command.

More detailed installation instructions can be found in the
[online documentation](https://miguelzuma.github.io/glow/usage.html#installation).

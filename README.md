# GLoW: Gravitational Lensing of Waves

If you find this code useful in your research, please cite the main GLoW paper
[Villarrubia-Rojo+ 24](https://inspirehep.net/literature/2826315).

In addition, GLoW has also been used in the following works:
* [Savastano+ 23](https://inspirehep.net/literature/2667175)
* [Zumalacarregui 24](https://inspirehep.net/literature/2781293)
* [Brando+ 24](https://inspirehep.net/literature/2804868)
* [Singh+ 25](https://inspirehep.net/literature/2885963)
* [Abe+ 25](https://inspirehep.net/literature/2931895)
* [Yuan+ 25](https://inspirehep.net/literature/2966083)
* [Vujeva+ 25](https://inspirehep.net/literature/3070503)
* [Caldarola+ 25](https://inspirehep.net/literature/3081783)
* [Goyal+ 25](https://inspirehep.net/literature/3094475)
* [Shan+ 25](https://inspirehep.net/literature/3094856)

The online documentation can be found
[here](https://miguelzuma.github.io/GLoW_public/index.html).

## Installation

![GLoW-Light](./sphinx_doc/diagrams/diagram_simp.png#gh-light-mode-only)
![GLoW-Dark](./sphinx_doc/diagrams/diagram_simp_dark.png##gh-dark-mode-only)

The pure Python version of the code should work out of the box. It only requires standard scientific
packages like ``numpy`` and ``scipy``.

The C version requires an external library, the GNU Scientific Library (GSL), that can be easily
installed with your favorite package manager. Alternatively, if you are using Conda, you can install
and activate the environment that we provide
```console
conda env create --file glow_env.yml && conda activate glow_env
```
Once the previous requirements are met, the code can be easily installed by running
```console
pip install .
```
in the main GLoW directory. If any error occurs, the file ``configure.log`` will contain additional
information. Open MP is also used to run certain parts of the code in parallel, but it is not
mandatory. If it is not correctly set up, the installation will configure the code in serial mode.

More detailed installation instructions can be found in the
[online documentation](https://miguelzuma.github.io/GLoW_public/usage.html#installation).

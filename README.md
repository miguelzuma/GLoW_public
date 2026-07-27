# GLoW: Gravitational Lensing of Waves

![GLoW logo](./sphinx_doc/diagrams/glow_logo.png#gh-light-mode-only)
![GLoW logo](./sphinx_doc/diagrams/glow_logo_dark.png#gh-dark-mode-only)

If you use this code, please cite this repository and the main GLoW paper
[Villarrubia-Rojo+ 24](https://inspirehep.net/literature/2826315).

The online documentation can be found
[here](https://miguelzuma.github.io/GLoW_public/index.html).

In addition, GLoW has also been used in the following works:
1. [Savastano+ 23](https://inspirehep.net/literature/2667175)
2. [Zumalacarregui 24](https://inspirehep.net/literature/2781293)
3. [Brando+ 24](https://inspirehep.net/literature/2804868)
4. [Singh+ 25](https://inspirehep.net/literature/2885963)
5. [Abe+ 25](https://inspirehep.net/literature/2931895)
6. [Yuan+ 25](https://inspirehep.net/literature/2966083)
7. [Vujeva+ 25](https://inspirehep.net/literature/3070503)
8. [Sun+ 25](https://arxiv.org/abs/2511.09107)
9. [Caldarola+ 25](https://inspirehep.net/literature/3081783)
10. [Goyal+ 25](https://inspirehep.net/literature/3094475)
11. [Shan+ 25](https://inspirehep.net/literature/3094856)
12. [Ando 26a](https://inspirehep.net/literature/3125747)
13. [Ephremidze+ 26](https://inspirehep.net/literature/3129087)
14. [Sun+ 26](https://arxiv.org/abs/2604.13930)
15. [Zumalacarregui & Shan 26](https://inspirehep.net/literature/3169257)
16. [Ando 26b](https://inspirehep.net/literature/3170969)
17. [Choi+ 26](https://inspirehep.net/literature/3180014)
18. [Cheung+ 26](https://inspirehep.net/literature/3183174)

## Installation

![GLoW-Light](./sphinx_doc/diagrams/diagram_simp.png#gh-light-mode-only)
![GLoW-Dark](./sphinx_doc/diagrams/diagram_simp_dark.png#gh-dark-mode-only)

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

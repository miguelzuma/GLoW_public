#### For PIP installation: 

(Recommended) Use latest version [miniconda3](https://docs.conda.io/projects/miniconda/en/latest/) to create an environment. 

1. Create new environment with conda (use below for python 3.11 on linux) and activate it:
   

`conda create --name easyglow python=3.11.6 cython=0.29.34 numpy mpmath`

If gives error try : `conda config --set channel_priority flexible` and rerun.

Note: Lower python versions may also work.

`conda activate easyglow`


2. Install GLoW:
Clone and enter the repository (glow directory in main branch) and then
   
`pip install .`

### Test installation: 
`python tests/test_lenses.py`
Run notebooks/examples_

**Note** : If you installed using make file before first execute: `make clean` and then go for pip installation.



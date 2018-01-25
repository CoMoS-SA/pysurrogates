# Fitting and evaluating  metamodels
==================================

The aim of this project is to provide researchers with a set of tools to fit surrogates (metamodels) to their models.
By 'models' here we refer to simulations on discrete timesteps which can be encoded in a python function. Models take a set of parameter values as input and gived out any kind of output. A calibration measure is necessary converts output of any dimensionality into a one dimentional variable.

*************
Installation:
*************

#########
- Windows
#########
**Clone the repository**

You might need to install git (`instructions <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_)
once you're done, open a Git Bash console (you should find it among your programs) and do:

``$git clone https://github.com/matuteiglesias/pysurrogates``

**Install**

You may need to install python, get it in `here <https://www.python.org/downloads/release/python-363/>`_. You can follow instructions `here <https://www.howtogeek.com/197947/how-to-install-python-on-windows/>`_. If successful, you have also installed pip. Open a console ('Win+R', type 'cmd' and press Enter), go to the directory where you cloned the repo, i.e.

``>cd path-to/pysurrogates/``
``>pip install .``

#######
- Linux
#######
You may need to install git

``$sudo apt-get install git-all``

**clone the repository**

``$git clone https://github.com/matuteiglesias/pysurrogates``
``$cd pysurrogates``

**install**

``$sudo pip install .``

You may need to get pip, follow instructions `here <https://pip.pypa.io/en/stable/installing/>`_ 

*********
Examples:
*********

Now you can open python and begin trying out the examples, i.e.

``>python``

``>>>import pysurrogates``

- See ipython notebooks for examples on what you can do.



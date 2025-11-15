Overview
========

Ways to use mrx
---------------

MRX is a collection of classes and functions that can be used in
several ways for optimizing 2D and 3D MHD equilibria without the 
assumption of nested flux surfaces. You can manipulate the objects
interactively, at the python command line or in a Jupyter notebook.

Input files
-----------

MRX problems are specified using a
python driver script, in which objects are defined and
configured. 

JAX and jit-compilation
-----------------------

MRX is written in a way that is compatible with JAX and jit-compilation 
for maximal performance after an initial compilation.

Optimization
------------

To do optimization using MRX, there are four basic steps:

1. Define the problem by specifying the geometry, the boundary type, the PDE being solved, and the 
resolution of the finite element space.
2. Initialize a deRham sequence and assemble the FEM spaces.
3. Initialize the initial magnetic field guess and the state of the simulation.
4. Run the relaxation loop.
5. Post-process the results.

This pattern is evident in the examples in this documentation. Much of the default functionality 
is available through mrx.utils. You can see the functionality used in the scripts/config_scripts folder.
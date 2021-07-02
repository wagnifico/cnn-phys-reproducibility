# Simulation of acoustic Gaussian pulses

## Dataset generation

This folder contains a C++ script to generate the 2D acoustic pulses databases.
It relies on the open-source Lattice-Boltzmann Fluids Mechanics library (University of Geneva). Please download the
library [here](https://gitlab.com/unigespc/palabos).

Compiling the code 
------------

On a UNIX system, execute the following commands to compile the solver :

<pre style="background-color:#cff0fa"><code>mkdir build/
cd build/
cmake ..
make
</code></pre>

Build folder and executable are ignored by the repository.

Current implementation considers that the Palabos folder is on the root directory. If not the case, replace the following line in `CMakeLists.txt`:

<pre style="background-color:#cff0fa"><code>set(PALABOS_ROOT "./palabos")</code></pre>

by:

<pre style="background-color:#cff0fa"><code>set(PALABOS_ROOT "/your/palabos/folder/")</code></pre>


Generating the databases
------------

For performing the simulations, do the following command:

<pre style="background-color:#cff0fa"><code>mpirun -np &#60;NTASKS&#62; acoustic_pulse &#60;paramXML&#62; </code></pre>

where `NTASKS` is the number of MPI processes used for the simulations and `paramXML` is an XML file with the simulation properties.
One can create a custom parameters input file. Present examples will reproduce the LBM properties and the database proportions used in the article:

+ `generate_database_train.xml` - training database, with 400 simulations of random pulses (1 to 4)
+ `generate_database_valid.xml` - validation database, with 100 simulations of random pulses (1 to 4)
+ `generate_database_test.xml` - testing database, with 100 simulations of random pulses (1 to 4)
+ `generate_database_benchmark_pulse.xml` - Gaussian pulse benchmark on the center of the domain
+ `generate_database_benchmark_opposed_gaussian.xml` - opposed Gaussians benchmark at the horizontal symmetry axis of the domain

Software versions
------------

[GCC](https://gcc.gnu.org/) 8.2.0, [OpenMPI](www.open-mpi.org) 4.0.0, [CMake](https://cmake.org/) 3.13.2 and [Palabos](https://palabos.unige.ch/) v2.2.1

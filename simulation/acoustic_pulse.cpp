/* 
 * Code to simulate the propagation of acoustic pulses in a closed rectangular
 * domain, reflecting boundary conditions.
 * 
 * This file is based on the Palabos library.
 *
 * The most recent release of Palabos can be downloaded at 
 * <http://www.palabos.org/>
 *
 * The library Palabos is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * The library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "palabos2D.h"
#include "palabos2D.hh"

#include <vector>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <random>
#include <sys/stat.h> // To create output folder.

typedef double T;
#define DESCRIPTOR plb::descriptors::D2Q9Descriptor

struct Param
{
    // User defined parameters
    // Geometry.
    T lx; // Size of computational domain, in physical units.
    T ly;
    // reference density
    T rho0;

    // Gaussian pulses (Defined by user).
    // Number of pulses
    plb::plint numUserPulses;
    // Position of the pulse center, in physical units.
    std::vector<T> userCx; 
    std::vector<T> userCy;
    // Gaussian pulse amplitude with respect to the base density
    std::vector<T> userEps;
    // Gaussian pulse half-width, in physical units. 
    std::vector<T> userB;
    
    // Randomly generated gaussian pulses
    // Number of pulses.
    plb::Array<plb::plint, 2> statNumRandPulses;
    // Position of the pulse center, in physical units
    plb::Array<T,2> statCx;
    plb::Array<T,2> statCy;
    // Gaussian pulse amplitude with respect to offset
    plb::Array<T,2> statEps;
    // Gaussian pulse half-width, in physical units
    plb::Array<T,2> statB;

    // Numerics
    // Kinematic viscosity, in physical units
    T nu; 
    T inletVelocity;
    // Velocity in lattice units (numerical parameters).
    T uLB;
    // Number of lattice nodes along a reference length.
    plb::plint resolution;
    // Output
    // Time for events in lattice units.
    T maxT; 

    // Derived parameters
    // Discrete space and time step.
    T dx; 
    T dt;
    // Kinematic viscosity, in lattice units.
    T nuLB;
    // Relaxation parameter.
    T omega;
    // Grid resolution of bounding box.
    plb::plint nx;
    plb::plint ny;

    // Pulses
    plb::plint numRandPulses;
    plb::plint totalNumPulses;
    // Position of the pulse center, in physical units.
    std::vector<T> cx;
    std::vector<T> cy;
    // Gaussian pulse amplitude with respect to offset. 
    std::vector<T> eps;
    // Gaussian pulse half-width, in physical units. 
    std::vector<T> b; 

    // Position of the center, in lattice units.
    std::vector<plb::plint> cxLB;
    std::vector<plb::plint> cyLB;
    
    // Output Directory
    std::string outputDir;
    // Time for events in lattice units.
    plb::plint maxIter;
    // Time for vtk output.
    plb::plint vtkIter;
    // Number of simulations.
    plb::plint numSim; 

    Param()
    {}

    Param(std::string xmlFname)
    {
	    plb::XMLreader document(xmlFname);
        // reading the xml input file
        document["geometry"]["lx"].read(lx);
        document["geometry"]["ly"].read(ly);   
        document["acoustics"]["rho0"].read(rho0);
        document["acoustics"]["userDefined"]["numUserPulses"].read(numUserPulses);
        PLB_ASSERT(numUserPulses >= 0);
        
        if (numUserPulses > 0) 
        {
            std::vector<T> x, y, k, l;
            document["acoustics"]["userDefined"]["center"]["x"].read(x);
            document["acoustics"]["userDefined"]["center"]["y"].read(y);
            document["acoustics"]["userDefined"]["amplitude"].read(k);
            document["acoustics"]["userDefined"]["halfWidth"].read(l);
            PLB_ASSERT(x.size() == numUserPulses && y.size() == numUserPulses);
            PLB_ASSERT(k.size() == numUserPulses && l.size() == numUserPulses);
            userCx.resize(numUserPulses);
            userCy.resize(numUserPulses);
            userEps.resize(numUserPulses);
            userB.resize(numUserPulses);
            for (plb::plint i = 0; i < numUserPulses; i++) {
                userCx[i] = x[i];
                userCy[i] = y[i];
                userEps[i] = k[i];
                userB[i] = l[i];
            }
        }

        std::vector<plb::plint> i;
        std::vector<T> u, v, x, y;
        document["acoustics"]["randomDefined"]["numRandPulses"].read(i);
        document["acoustics"]["randomDefined"]["center"]["x"].read(u);
        document["acoustics"]["randomDefined"]["center"]["y"].read(v);
        document["acoustics"]["randomDefined"]["amplitude"].read(x);
        document["acoustics"]["randomDefined"]["halfWidth"].read(y);

        for (plb::plint j = 0; j < 2; j++)
        {
            statNumRandPulses[j] = i[j];
            statCx[j] = u[j];
            statCy[j] = v[j];
            statEps[j] = x[j];
            statB[j] = y[j];
        }
        
        document["numerics"]["nu"].read(nu);
        document["numerics"]["inletVelocity"].read(inletVelocity);
        document["numerics"]["uLB"].read(uLB);
        document["numerics"]["resolution"].read(resolution);      

        document["output"]["outputDir"].read(outputDir);
        document["output"]["maxT"].read(maxT);
        document["output"]["vtkIter"].read(vtkIter);
        document["output"]["numSim"].read(numSim);
    }
};


void setRandomPulses(Param& param, std::mt19937& gen,
        std::uniform_int_distribution<plb::plint>& distNum,
        std::uniform_real_distribution<T>& distCx,
        std::uniform_real_distribution<T>& distCy,
        std::uniform_real_distribution<T>& distEps,
        std::uniform_real_distribution<T>& distB )
{
    param.numRandPulses = distNum(gen);
    param.totalNumPulses = param.numRandPulses + param.numUserPulses;

    plb::global::mpi().bCast(&param.numRandPulses, 1);
    plb::global::mpi().bCast(&param.totalNumPulses, 1);
    
    std::vector<T> tmpCx(param.numRandPulses);                               
    std::vector<T> tmpCy(param.numRandPulses);                               
    std::vector<T> tmpEps(param.numRandPulses);                               
    std::vector<T> tmpB(param.numRandPulses);   

    param.cx = param.userCx;
    param.cy = param.userCy;
    param.eps = param.userEps;
    param.b = param.userB;

    if (plb::global::mpi().isMainProcessor())
    {
        for (plb::plint i = 0; i < param.numRandPulses; i++) {
            tmpCx[i] = distCx(gen);
            tmpCy[i] = distCy(gen);
            tmpEps[i] = distEps(gen);
            tmpB[i] = distB(gen);
        }
    }

    for (plb::plint i = param.numUserPulses; i < param.totalNumPulses; i++)
    {
        plb::global::mpi().bCast(&tmpCx[i], 1);
        plb::global::mpi().bCast(&tmpCy[i], 1);
        plb::global::mpi().bCast(&tmpEps[i],1);
        plb::global::mpi().bCast(&tmpB[i],  1);
    }

    param.cx.insert(std::end(param.cx), std::begin(tmpCx), std::end(tmpCx)); 
    param.cy.insert(std::end(param.cy), std::begin(tmpCy), std::end(tmpCy)); 
    param.eps.insert(std::end(param.eps), std::begin(tmpEps), std::end(tmpEps));
    param.b.insert(std::end(param.b), std::begin(tmpB), std::end(tmpB));
}

int mkpath(const char* file_path, mode_t mode)
{
    assert(file_path && *file_path);
    char* p;
    char* fp;
    fp = const_cast<char*>(file_path);
    for (p=strchr(fp+1, '/'); p; p=strchr(p+1, '/')) {
    *p='\0';
    if (mkdir(fp, mode)==-1)
    {
        if (errno!=EEXIST) {
            *p='/'; return -1;
        }
    }
    *p='/';
    }
    return 0;
}

std::string format_account_number(int acct_no)
{
    char buffer[7];
    std::snprintf(buffer, sizeof(buffer), "%06d", acct_no);
    return buffer;
}

void computeDerivedParameters(Param& param)
{
    // discrete space step
    param.dx = param.ly / (param.resolution - 1.0); 
    param.dt = (param.uLB/param.inletVelocity) * param.dx;
    // kinematic viscosity, in lattice units
    param.nuLB = param.nu * param.dt/(param.dx*param.dx);
    // relaxation parameter
    param.omega = 1.0/(DESCRIPTOR<T>::invCs2*param.nuLB+0.5);    
    // grid resolution of bounding box
    param.nx = plb::util::roundToInt(param.lx/param.dx) + 1;          
    param.ny = plb::util::roundToInt(param.ly/param.dx) + 1;
    
    param.cxLB.resize(param.totalNumPulses);
    param.cyLB.resize(param.totalNumPulses);
    for (plb::plint i = 0; i < param.totalNumPulses; i++)
    {
        // position of the center, in lattice units
        param.cxLB[i] = plb::util::roundToInt(param.cx[i]/param.dx);  
        param.cyLB[i] = plb::util::roundToInt(param.cy[i]/param.dx);
    }    
    // time for events in lattice units
    param.maxIter = plb::util::roundToInt(param.maxT/param.dt);       
}

void printParam(Param& param)
{
    plb::pcout << "User defined parameters: " <<std::endl;
    plb::pcout << "lx = " << param.lx << std::endl;
    plb::pcout << "ly = " << param.ly << std::endl << std::endl;

    plb::pcout << "INITIAL CONDITIONS" << std::endl;
    plb::pcout << "Number of acoustic pulses: "
        << param.totalNumPulses << std::endl << std::endl;

    for (plb::plint i = 0; i < param.totalNumPulses; i++)
    {
        plb::pcout << "Pulse (" << i << "): " << std::endl;
        plb::pcout << "cx = " << param.cx[i] << std::endl;
        plb::pcout << "cy = " << param.cy[i]<< std::endl;
        plb::pcout << "eps = " << param.eps[i] << std::endl;
        plb::pcout << "b = " << param.b[i] << std::endl << std::endl;
    }

    plb::pcout << "nu = " << param.nu << std::endl;
    plb::pcout << "vel = " << param.inletVelocity << std::endl;
    plb::pcout << "uLB = " << param.uLB << std::endl;
    plb::pcout << "res = " << param.resolution << std::endl;
    
    plb::pcout << "maxT = " << param.maxT << std::endl << std::endl;
    
    plb::pcout << "Derived parameters: " << std::endl;
    plb::pcout << "dx = " << param.dx << std::endl;
    plb::pcout << "dt = " << param.dt << std::endl;
    plb::pcout << "nuLB = " << param.nuLB << std::endl;
    plb::pcout << "omega = " << param.omega << std::endl;
    plb::pcout << "nx = " << param.nx << std::endl;
    plb::pcout << "ny = " << param.ny << std::endl;
}

Param param;

plb::Array<T,2> u0(0,0);

// Function to calculate the initial density from pulses amplitude and
// location
void getInitialDensity(
    plb::plint iX, plb::plint iY, T& rho, plb::Array<T,2>& u)
{
    u = u0;
    T x = (T)iX * param.dx;
    T y = (T)iY * param.dx;
    
    rho = param.rho0;
    
    for (plb::plint i = 0; i < param.totalNumPulses; i++)
    {
        T alp = log(2.)/(param.b[i]*param.b[i]);
        rho += (param.eps[i]*(exp ((-alp*1.)*(
                    (x-param.cx[i])*(x-param.cx[i]) 
                        + (y-param.cy[i])*(y-param.cy[i])
                        ) 
                    )
                )
            );
    }
}

// Initialize the lattice at zero velocity and the initial density field.
void defineInitialDensity(
	plb::MultiBlockLattice2D<T,DESCRIPTOR>& lattice)
{
    // Create the initial condition.
    initializeAtEquilibrium(
           lattice, lattice.getBoundingBox(), getInitialDensity
    );
    lattice.initialize();
}

void createZones(plb::MultiBlockLattice2D<T,DESCRIPTOR>& lattice)
{
    plb::plint nx = param.nx;
    plb::plint ny = param.ny;
    plb::Box2D fullBox(0, nx-1, 0, ny-1);
}

// Function to export the fields in vtk format
void writeVTK(plb::MultiBlockLattice2D<T,DESCRIPTOR>& lattice,
              plb::plint iter)
{
    T dx = param.dx;
    T dt = param.dt;
    plb::VtkImageOutput2D<T> vtkOut(plb::createFileName("vtk", iter, 6), dx);
    vtkOut.writeData<double>(*computeDensity(lattice), "density", 1.);
    vtkOut.writeData<2,double>(*computeVelocity(lattice), "velocity", dx/dt);
}

int main(int argc, char* argv[])
{
    plb::plbInit(&argc, &argv);
    std::string xmlFileName;
    
    try
    {
        plb::global::argv(1).read(xmlFileName);
    }
    catch (plb::PlbIOException& exception)
    {
        plb::pcout << "Wrong parameters; the syntax is: "
            << (std::string)plb::global::argv(0)
                << " input-file.xml" << std::endl;
        return -1;
    }

    // reading the input file (.xml)
    try
    {
        param = Param(xmlFileName);
    }
    catch (plb::PlbIOException& exception)
    {
        plb::pcout << exception.what() << std::endl;
        return -1;
    }
    plb::global::IOpolicy().activateParallelIO(true);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<plb::plint> uniformPulses(
        param.statNumRandPulses[0], param.statNumRandPulses[1]);
    std::uniform_real_distribution<T> uniformCx(
        param.statCx[0], param.statCx[1]);
    std::uniform_real_distribution<T> uniformCy(
        param.statCy[0], param.statCy[1]);
    std::uniform_real_distribution<T> uniformEps(
        param.statEps[0], param.statEps[1]);
    std::uniform_real_distribution<T> uniformB(
        param.statB[0], param.statB[1]);

    // Loop over number of simulations (new initial conditions)
    for(plb::plint sim=0; sim < param.numSim;  ++sim)
    {        	
	    // creating the data folder
        std::string tmpOutDir = param.outputDir + '/' + format_account_number(sim) + '/';
        if (plb::global::mpi().isMainProcessor())
        {
            mkpath(tmpOutDir.c_str(), 0755);
        }
        
	    // generating the random pulse
        setRandomPulses(param, gen, uniformPulses, uniformCx, uniformCy,
                uniformEps, uniformB);
        
        computeDerivedParameters(param);
        printParam(param);
        plb::global::directories().setOutputDir(tmpOutDir + "/");
            
        plb::pcout << "Initialising..." << std::endl;

        // The default constructor creates only one "dynamic" for all lattice points. In order to
        // vary omega (sponge zones) it is necessary to assign a dynamic to each point.
        plb::Dynamics<T,DESCRIPTOR> *dynamics = new plb::BGKdynamics<T,DESCRIPTOR>(param.omega);
        plb::MultiBlockLattice2D<T, DESCRIPTOR> lattice (
            param.nx, param.ny, new plb::BGKdynamics<T,DESCRIPTOR>(param.omega)
            );
            
        defineDynamics(lattice, lattice.getBoundingBox(), dynamics->clone());
            delete dynamics;
            lattice.toggleInternalStatistics(false);

        plb::OnLatticeBoundaryCondition2D<T,DESCRIPTOR> *bc = plb::createLocalBoundaryCondition2D<T,DESCRIPTOR>();
        plb::pcout << "Generating outer domain boundary conditions." << std::endl;
        lattice.periodicity().toggleAll(false);
        bc -> setVelocityConditionOnBlockBoundaries(lattice);

        delete bc; bc = 0;
   
        plb::pcout << "Initialisation finished!" << std::endl;
        
        // starting the density based on the pulse(s)
        defineInitialDensity(lattice);
        
        // Main loop over time iterations.
        for (plb::plint iT=0; iT<param.maxIter; ++iT)
        {
            // Execute lattice Boltzmann iteration.
            lattice.collideAndStream();
            
            if (iT% param.vtkIter ==0) {       
                plb::pcout << "Iteration: " << iT << std::endl;
                // writing the result in a VTK file
                writeVTK(lattice, iT);
            }
        }

        param.cx.clear();
        param.cy.clear();
        param.eps.clear();
        param.b.clear();
      
        plb::pcout << "End of run " << sim << std::endl << std::endl;

    }

}

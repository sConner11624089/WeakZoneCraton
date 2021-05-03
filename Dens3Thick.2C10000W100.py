#!/usr/bin/env python
# coding: utf-8

# Simple Craton Case with Weakzones
# ----------
# 
# Let's start simple and put a craton with a constant viscosity into the temperature field of a well-developed isoviscous convecting mantle.  Note, be sure that the temperature & mesh files that you load correspond to the same Ra, mesh size, and resolution as what you want to use for the craton models.

# First, let's tell Python what it needs:

# In[1]:


import underworld as uw
from underworld import function as fn
import math
import time as timekeeper
import numpy
import glucifer
import matplotlib.pyplot as plt
from IPython import display
uw.matplotlib_inline()

rank = uw.rank()
#rank = uw.mpi.rank


# Set up parameters of model space
# ------
# 

# In[2]:


# Set simulation box size.
boxHeight = 1.0
boxLength = 3.0
# Set the resolution.
res = 128   # make sure this resolution matches what you eventually will use for the other model.  
            # Otherwise you'll have to play some extrapolation tricks 
# Set min/max temperatures.
tempMin = 0.0
tempMax = 1.0


# Set up mesh
# -----------

# In[3]:


mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                 elementRes  = (int(boxLength*res), res), 
                                 minCoord    = (0., 0.), 
                                 maxCoord    = (boxLength, boxHeight),
                                 periodic     = [True, False])

velocityField       = mesh.add_variable(         nodeDofCount=2 )
pressureField       = mesh.subMesh.add_variable( nodeDofCount=1 )
temperatureField    = mesh.add_variable(         nodeDofCount=1 )
temperatureDotField = mesh.add_variable(         nodeDofCount=1 )

# Initialise values
velocityField.data[:]       = [0.,0.]
pressureField.data[:]       = 0.


# Let's Load Data from other model
# -------
# 

# In[4]:


# Read temperature data
readTemperature = True
# Read swarm data
loadData = True

#determining the last step ran


import os
try:
    workdir
except NameError:
    workdir = os.path.abspath(".")
    
dir_output = os.path.join(workdir,"NewTempData/")


if not loadData:
    step = 0
    time = 0.0
    rStep = -1.0
else: 
    dataload = numpy.loadtxt(dir_output + 'FrequentOutput.dat', skiprows=4)
    nL = dataload[-1,0]
    nL = int(-1-(nL % 1000))
    step = int(dataload[nL,0])
    time = dataload[nL,1] 
    rStep = step

#print('Starting at step %i and time %.2E' %(step,time))


if readTemperature:
    temperatureField.load(dir_output + 'temperature_%i.h5' %step, interpolate=True)

else:  #this will set up a sinusoidal temp field
    pertStrength = 0.2
    deltaTemp = tempMax - tempMin
    for index, coord in enumerate(mesh.data):
        pertCoeff = math.cos( math.pi * coord[0] ) * math.sin( math.pi * coord[1] )
        temperatureField.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
        temperatureField.data[index] = max(tempMin, min(tempMax, temperatureField.data[index]))


# Boundary & Initial Conditions
# ------

# Set top and bottom wall temperature boundary values.

# In[5]:


for index in mesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = tempMax
for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = tempMin


# In[6]:


iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]

BottomWall = mesh.specialSets["MinJ_VertexSet"] 
TopWall = mesh.specialSets["MaxJ_VertexSet"] 
LeftWall = mesh.specialSets["MinI_VertexSet"] 
RightWall = mesh.specialSets["MaxI_VertexSet"] 


# Construct sets for ``I`` (vertical) and ``J`` (horizontal) walls.

# Create Direchlet, or fixed value, boundary conditions. More information on setting boundary conditions can be found in the **Systems** section of the user guide.

# In[7]:


# 2D velocity vector can have two Dirichlet conditions on each vertex, 
# v_x is fixed on the iWalls (vertical), v_y is fixed on the jWalls (horizontal) - freeslip on all sides

# make sure these match the boundary conditions you'll eventually use for the full model

velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
                                           indexSetsPerDof = (iWalls, jWalls) )

# Temperature is held constant on the jWalls
tempBC = uw.conditions.DirichletCondition( variable        = temperatureField, 
                                           indexSetsPerDof = (jWalls,) )


# Let's see if we got it correct...

# In[8]:


figtemp = glucifer.Figure( figsize=(800,400) )   #Needs to be commented when sending to Kamiak - uncomment when you are using notebook
figtemp.append( glucifer.objects.Surface(mesh, temperatureField, colours="blue white red") )
#figtemp.append( glucifer.objects.Mesh(mesh) )
#figtemp.show()

temperatureFieldIntegral = uw.utils.Integral(fn = temperatureField,mesh= mesh,integrationType="volume")
volume_integral = uw.utils.Integral( mesh=mesh, fn=1., integrationType="volume" )
volume = volume_integral.evaluate()
avTemperature = temperatureFieldIntegral.evaluate()[0]/volume[0]
#print (avTemperature)


# Set up material parameters and functions
# ----------
# 
# Set functions for viscosity, density and buoyancy force. These functions and variables only need to be defined at the beginning of the simulation, not each timestep.

# Starting with Materials and Swarm
# ------

# In[9]:


swarm         = uw.swarm.Swarm( mesh=mesh )
swarmLayout      = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
#swarmLayout      = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
swarm.populate_using_layout( layout=swarmLayout )

nParticles = 20

# particle population control object (has to be called)
population_control = uw.swarm.PopulationControl(swarm,
                                                aggressive=False,splitThreshold=0.15, maxDeletions=2,maxSplits=5,
                                                particlesPerCell=nParticles)


# Set up material swarms
materialIndex  = swarm.add_variable( dataType="int",    count=1 )

# all the potential materials.  for the reference case, let's just have average crustal structure
continentalMaterial = 0
weakMaterial = 1
refmantleMaterial = 2

continentalDepth = 0.7
contX1 =  1.00       #use these when we want to embed continents of a certain size rather than across entire model
contX2 =  2.00

weakzoneDepth = 0.7
weakzoneX1 = 1.6
weakzoneX2 = 1.8

#give shapes a material

materialIndex.data[:] = refmantleMaterial
for index,coord in enumerate(swarm.particleCoordinates.data):
     
    if coord[1] > continentalDepth+0.05 and coord[0] > contX1 and coord[0] < contX1+.1:
        materialIndex.data[index] = continentalMaterial
   
    if coord[1] > continentalDepth - coord[0]*.5+.6 and coord[0] > contX1+.1 and coord[0] < contX1+.2:
        materialIndex.data[index] = continentalMaterial
   
    if coord[1] > continentalDepth and coord[0] > contX1+.2 and coord[0] < contX1+.8:
        materialIndex.data[index] = continentalMaterial
       
    if coord[1] > weakzoneDepth and coord[0] > weakzoneX1 and coord[0] < weakzoneX2:
        materialIndex.data[index] = weakMaterial
   
    if coord[1] > continentalDepth + coord[0]*.5-.9 and coord[0] > contX1+.8 and coord[0] < contX2-.1:
        materialIndex.data[index] = continentalMaterial
   
    if coord[1] > continentalDepth+0.05 and coord[0] > contX2-.1 and coord[0] < contX2:
        materialIndex.data[index] = continentalMaterial

materialPoints = glucifer.objects.Points(swarm, materialIndex, pointSize=3.,  colours='purple red green blue gray')
materialPoints.colourBar.properties = {"ticks" : 2, "margin" : 40, "width" : 10, "align" : "center"}

#comment for Kamiak, uncomment for notebook

figMaterialMesh = glucifer.Figure(title="Materials and Mesh", quality=3)
#figMaterialMesh.append( glucifer.objects.Mesh(mesh) )
figMaterialMesh.append( materialPoints )
#figMaterialMesh.show() 
        
        
# we'll assign values to the materials like rheology, density, etc, in a bit


# Setting Values to Materials
# -----

# In[10]:


#Arrhenius viscosity
#eta0 = 1.0e-6
#activationEnergy = 27.63102112
#fn_viscosity = eta0 * fn.math.exp( activationEnergy / (temperatureField+1.) )

#F-K approximation

tempdepend = False #toggle to use temp depend viscosity

if tempdepend :
    surfEtaCont = 1.0e1    #highest viscosity for continents
    surfEtaWeak = 1.0e1
    surfEtaMantle = 1.0e0  #highest viscosity for mantle
    cEtaCont = numpy.log(surfEtaCont) / tempMax
    cEtaWeak = numpy.log(surfEtaWeak) /tempMax
    cEtaMantle = numpy.log(surfEtaMantle) / tempMax

else :
    cEtaCont = 1e4
    cEtaWeak = 100.0
    cEtaMantle = 1.0


refcEtaMap  = {      continentalMaterial : cEtaCont, 
                     weakMaterial : cEtaWeak,
                     refmantleMaterial : cEtaMantle }

refcEtaFn    = fn.branching.map( fn_key = materialIndex, mapping = refcEtaMap )

if tempdepend :
    fn_viscosity = uw.function.math.exp(refcEtaFn *(tempMax-temperatureField))

else :   
    fn_viscosity = refcEtaFn 


figEta = glucifer.Figure(title="Viscosity", quality=3)
figEta.append ( glucifer.objects.Points(swarm,fn_colour = fn_viscosity, fn_size=7 ))
#figEta.show() 


#density

densCont = -0.885
densWeak = - 0.885
densMantle = 0.0

refDensMap = {       continentalMaterial: densCont,
                     weakMaterial: densWeak, 
                     refmantleMaterial: densMantle}

# Density & Buoyancy Functions
# --

# 
# $$
#     Ra = \frac{\alpha\rho g \Delta T h^3}{\kappa \eta_{ref}}   ;   Rb = \frac{ \Delta\rho g h^3}{\kappa\eta_{ref}}
# $$
# 

# In[11]:


# Rayleigh number.
Ra = 1.0e7  # make sure this matches what you used in your start-up models.  also, watch your resolution if you set this higher

Rb = 1.0e7  #sets up buoyancy scheme  

# Define our vertical unit vector using a python tuple (this will be automatically converted to a function).
z_hat = ( 0.0, 1.0 )

contbuoy = True # set this to true if you plan on using different densities for the continental material

if contbuoy:
    # construct the density function using material properties outlined above
    densityFn = fn.branching.map( fn_key = materialIndex, mapping = refDensMap )
    # creating a buoyancy force vector
    buoyancyFn = (Ra * temperatureField - Rb * densityFn)  * z_hat
    
else:
    # Construct our density function.
    densityFn = Ra * temperatureField
    # Now create a buoyancy force vector using the density and the vertical unit vector. 
    buoyancyFn = densityFn * z_hat


# Bookkeeping
# -----
# Where should we keep our results?

# In[12]:


# Make output directory if necessary

import os
try:
    workdir
except NameError:
    workdir = os.path.abspath(".")

outputPath = os.path.join(workdir,"D3T.2C10000W100x/")

if rank==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)  
    
writefigures = True  #toggle to set whether to write figures to output directory
# Output model timestep info

if  rank==0:
    start = timekeeper.time()

    fw = open(outputPath + 'FrequentOutput.dat',"w")
    fw.write("%s \n" %(timekeeper.ctime()))
#   fw.write("Running on %i proc(s). \n" %size)  #only need to use if running on more than one processor
    fw.close()
    


# System Setup
# -------
# **Setup a Stokes system**
# 
# Underworld uses the Stokes system to solve the incompressible Stokes equations.  

# In[13]:


stokes = uw.systems.Stokes( velocityField = velocityField, 
                            pressureField = pressureField,
                            conditions    = velBC,
                            fn_viscosity  = fn_viscosity, 
                            fn_bodyforce  = buoyancyFn )

# get the default stokes equation solver
solver = uw.systems.Solver( stokes )


# **Set up the advective diffusive system**
# 
# Underworld uses the AdvectionDiffusion system to solve the temperature field given heat transport through the velocity field.

# In[14]:


advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField, 
                                         fn_diffusivity = 1.0, 
                                         conditions     = tempBC )

# Create a system to advect the swarm YOU MUST USE THIS IF YOU USE SWARMS TO PUT IN MATERIALS
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )


# **Analysis Tools**

# **Nusselt number**
# 
# The Nusselt number is the ratio between convective and conductive heat transfer
# 
# \\[
# Nu = -h \frac{ \int_0^l \partial_z T (x, z=h) dx}{ \int_0^l T (x, z=0) dx}
# \\]
# 
# 
# 
# 

# In[15]:


nuTop    = uw.utils.Integral( fn=temperatureField.fn_gradient[1], 
                              mesh=mesh, integrationType='Surface', 
                              surfaceIndexSet=mesh.specialSets["MaxJ_VertexSet"])

nuBottom = uw.utils.Integral( fn=temperatureField,               
                              mesh=mesh, integrationType='Surface', 
                              surfaceIndexSet=mesh.specialSets["MinJ_VertexSet"])


# In[16]:


nu = - nuTop.evaluate()[0]/nuBottom.evaluate()[0]
#print('Nusselt number = {0:.6f}'.format(nu))


# **RMS velocity**
# 
# The root mean squared velocity is defined by intergrating over the entire simulation domain via
# 
# \\[
# \begin{aligned}
# v_{rms}  =  \sqrt{ \frac{ \int_V (\mathbf{v}.\mathbf{v}) dV } {\int_V dV} }
# \end{aligned}
# \\]
# 
# where $V$ denotes the volume of the box.

# In[17]:


intVdotV = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )

vrms = math.sqrt( intVdotV.evaluate()[0]/ volume [0] )
#print('Initial vrms = {0:.3f}'.format(vrms))


# **Heat Flow**

# In[18]:


# Integrals for calculating heat-flow - note these are the total heat flow (if dimensional would be Watts) across the surface NOT FLUXES

surfaceHF = uw.utils.Integral( fn = temperatureField.fn_gradient[1], mesh = mesh, integrationType = "surface", surfaceIndexSet = TopWall)
bottomHF = uw.utils.Integral( fn = temperatureField.fn_gradient[1], mesh = mesh, integrationType = "surface", surfaceIndexSet = BottomWall)
leftHF = uw.utils.Integral( fn = temperatureField.fn_gradient[0], mesh = mesh, integrationType = "surface", surfaceIndexSet = LeftWall)
rightHF = uw.utils.Integral( fn = temperatureField.fn_gradient[0], mesh = mesh, integrationType = "surface", surfaceIndexSet = RightWall)


# Main time stepping loop
# -----

# In[19]:


# init these guys

time = 0.
step = 0
steps_end = 5000
checkpointstep = 500
writestep = 500



# initalize values
vrms, nu = 0.0, 0.0
dt = min(advector.get_max_dt(), advDiff.get_max_dt())

if rank ==0:
        fw = open( outputPath + 'FrequentOutput.dat',"a")
        fw.write("Setup time: %.2f seconds\n" %(timekeeper.time() - start))
        fw.write("--------------------- \n")
        fw.write("Step \t Time \t Stopwatch \t Average Temperature \t Nusselt Number \t Vrms \t Surface Heat Flux \t Bottom Heat Flux \t Other Walls Heat Flux \n")
        start = timekeeper.time()
        fw.close()


if rank == 0:
      start = timekeeper.time() # Setup clock to calculate simulation CPU time.

trackHF = True

if trackHF:    
    arrMeanTemp = numpy.zeros(steps_end+1)
    arrSurfHF = numpy.zeros(steps_end+1) 
    arrOtherWallsHF = numpy.zeros(steps_end+1) 
    arrMaxTemp = numpy.zeros(steps_end+1) 
    arrNu = numpy.zeros(steps_end+1)
    arrVrms = numpy.zeros (steps_end+1)
    

# perform timestepping

while step < steps_end:
    # Solve for the velocity field given the current temperature field.
    solver.solve()
    dt = min(advector.get_max_dt(), advDiff.get_max_dt())
    advector.integrate(dt)
    advDiff.integrate(dt)
    time += dt
    step += 1
    avTemperature = temperatureFieldIntegral.evaluate()[0]/volume[0]
    vrms = math.sqrt( intVdotV.evaluate()[0] / volume[0])
    nu = - nuTop.evaluate()[0]/nuBottom.evaluate()[0]
            
    if trackHF:
        
        surfHF = -1. * surfaceHF.evaluate()[0]
        bottHF = -1. * bottomHF.evaluate()[0]
        wallsHF = abs(bottomHF.evaluate()[0])+abs(leftHF.evaluate()[0])+abs(rightHF.evaluate()[0])
        
        if rank == 0:
            arrMeanTemp[step] = avTemperature
            arrSurfHF[step] = surfHF
            arrOtherWallsHF[step] = wallsHF
            arrNu[step] = nu
            arrVrms[step] = vrms

    if rank==0:
        fw = open( outputPath  + 'FrequentOutput.dat',"a")
        fw.write("%i \t %.2f \t %.2f \t  %.5f \t %.5f \t %.5f \t %.5f \t %.5f \t %.5f \t \n" %(step, time, timekeeper.time() - start, avTemperature, nu, vrms, surfHF, bottHF, wallsHF ))
        start = timekeeper.time()
        fw.close()
        
    if step % checkpointstep == 0.:
        mesh.save(outputPath + 'mesh_%i.h5' %step)
        temperatureField.save(outputPath +'temperature_%i.h5' %step )
        velocityField.save(outputPath + 'velocity_%i.h5' %step )
        swarm.save(outputPath + 'swarm_%i.h5' %step)
        
    if step % writestep ==0. and writefigures:
        figtemp.save_image(outputPath + 'temperatureplot_%i' %step)
        figEta.save_image(outputPath + 'viscosityplot_%i' %step)
        figMaterialMesh.save_image(outputPath + 'materialsplot_%i' %step)
    
   # if step % writestep ==0. and writefigures and rank == 0: 
   #     n = 100
   #     topWallX = numpy.linspace(mesh.minCoord[0],mesh.maxCoord[0],n)
   #     topWallVelocity = numpy.zeros(n)
   #
   #     for i in range(n):
   #         topWallVelocity[i] = velocityField[0].evaluate_global((topWallX[i],mesh.maxCoord[1]))
   #     plt.clf()
   #     plt.plot(topWallX,topWallVelocity)
   #     plt.title("Surface Velocity")
   #     plt.ylabel("Horizontal Velocity")
   #     plt.xlabel("Distance")
   #     plt.savefig(outputPath + "%s_surface_velocity.pdf" %step)
        
    if step % 10.0 == 0.0:
        population_control.repopulate()
        


# **Plot final temperature, velocity field, and viscosity structure**

# In[20]:


# plot figure
#figtemp = glucifer.Figure( figsize=(800,400) )
#figtemp.append( glucifer.objects.Surface(mesh, temperatureField, colours="blue white red") )
#figtemp.append( glucifer.objects.VectorArrows(mesh, velocityField/10000.0, arrowHead=0.2, scaling=0.1) )
#figtemp.show()

#figEta.show()
#figMaterialMesh.show() 


# **Surface Velocity Plots**

# In[21]:


#n = 100
#topWallX = numpy.linspace(mesh.minCoord[0],mesh.maxCoord[0],n)
#topWallVelocity = numpy.zeros(n)

#for i in range(n):
#    topWallVelocity[i] = velocityField[0].evaluate_global((topWallX[i],mesh.maxCoord[1]))

#if rank == 0:
#    plt.clf()

#    plt.plot(topWallX,topWallVelocity)
#    plt.title("Surface Velocity")
#    plt.ylabel("Horizontal Velocity")
#    plt.xlabel("Distance")
#    if writefigures:
#       plt.savefig(outputPath + "%s_surface_velocity.pdf")


# **Time Series Plots**

# In[22]:


#if rank == 0 and trackHF:
#    plt.plot(range(steps_end),arrMeanTemp[:step])
#    plt.scatter(range(steps_end),arrMeanTemp[:step])

#    plt.xlabel("Time Step")
#    plt.ylabel("Average Temperature")
#if writefigures:
#        plt.savefig(outputPath +"%s_temperaturetime.pdf" , bbox_inches="tight")


# In[23]:



#if rank ==0:
#    plt.plot(range(steps_end),arrNu[:step])
#    plt.scatter(range(steps_end),arrNu[:step])
#    plt.xlabel("Time Step")
#    plt.ylabel("Nusselt Number")
    #plt.show()
    
#if writefigures:
#    plt.savefig(outputPath +"/%s_nusselttime.pdf" , bbox_inches="tight")


# In[24]:



#if rank ==0:
#    plt.plot(range(steps_end),arrVrms[:step])
#    plt.scatter(range(steps_end),arrVrms[:step])
#    plt.xlabel("Time Step")
#    plt.ylabel("Vrms")
    #plt.show()

#if writefigures:
#    plt.savefig(outputPath + "%s_vrmstime.pdf", bbox_inches="tight")


# In[ ]:





# -*- coding: utf-8 -*-
"""Simplistic version a complete ERT Modelling->Inversion example."""
import matplotlib as mpl
mpl.use('Qt4Agg')
import pygimli as pg
import pygimli.meshtools as plc  # save space

import pybert as pb

# Create geometry definition for the modelling domain.
# worldMarker=True indicates the default boundary conditions for the ERT
world = plc.createWorld(start=[-50, 0], end=[50, -50],
                        layers=[-1, -5], worldMarker=True)

# Create some heterogeneous circular
block = plc.createCircle(pos=[0, -3.], radius=1, marker=4,
                         boundaryMarker=10, area=0.1)

# Merge geometry definition into a Piecewise Linear Complex (PLC)
geom = plc.mergePLC([world, block])

# Optional: show the geometry
pg.show(geom)

# Create a Dipole Dipole ('dd') measuring scheme with 21 electrodes.
scheme = pb.createData(elecs=pg.utils.grange(start=-10, end=10, n=21),
                       schemeName='dd')

# Put all electrodes (aka. sensors positions) into the PLC to enforce mesh
# refinement. Due to experience known, its convenient to add further refinement
# nodes in a distance of 10% of electrode spacing, to achieve sufficient
# numerical accuracy.
for pos in scheme.sensorPositions():
    geom.createNode(pos)
    geom.createNode(pos + pg.RVector3(0, -0.1))

# Create a mesh for the finite element modelling with appropriate mesh quality.
mesh = plc.createMesh(geom, quality=34)

# Optional: take a look at the mesh
pg.show(mesh)

# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 100.],
          [2, 50.],
          [3, 10.],
          [4, 100.]]

# Initialize the ERTManager (The class name is a subject to further change!)
ert = pb.Resistivity()

# Perform the modeling with the mesh and the measuring scheme itself
# and return a data container with apparent resistivity values,
# geometric factors and estimated data errors specified by the noise setting.
# The noise is also added to the data.
data = ert.simulate(mesh, res=rhomap, scheme=scheme,
                    noiseLevel=2, noiseAbs=1e-5)

# Optional: you can filter all values and tokens in the data container.
print('Simulated rhoa', data('rhoa'), max(data('rhoa')))
data.markInvalid(data('rhoa') < 0)
print('Filtered rhoa', data('rhoa'), max(data('rhoa')))
data.removeInvalid()

# Optional: save the data for further use
data.save('simple.dat')

# Optional: take a look at the data
pb.show(data)

# Run the ERTManager to invert the modeled data.
# The necessary inversion mesh is generated automatic.
model = ert.invert(data, paraDX=0.3, maxCellArea=0.2, lam=30)

# Let the ERTManger show you the model and fitting results of the last
# successful run.
ert.showResultAndFit()

# Optional: provide a custom mesh to the inversion
grid = pg.createGrid(x=pg.utils.grange(start=-12, end=12, n=33),
                     y=pg.utils.grange(start=-8, end=0, n=16))
mesh = pg.meshtools.appendTriangleBoundary(grid, xbound=50, ybound=50)

model = ert.invert(data, mesh=mesh, lam=30)
ert.showResultAndFit()

# Stop the script here and wait until all figure are closed.
pg.wait()

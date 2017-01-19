#!/usr/bin/env python
import numpy as np

import pygimli as pg
import pygimli.meshtools as plc

import pybert as pb
from pybert.sip import SIPdata

# Create geometry definition for the modelling domain
world = plc.createWorld(start=[-50, 0], end=[50, -50],
                        layers=[-1, -5], worldMarker=True)
# Create some heterogeneous circle
block = plc.createCircle(pos=[0, -3.], radius=1, marker=4,
                         boundaryMarker=10, area=0.1)
# Merge geometry definition
geom = plc.mergePLC([world, block])
# create measuring scheme (data container without data)
scheme = pb.createData(elecs=np.linspace(-10, 10, 21),
                       schemeName='dd')
for pos in scheme.sensorPositions():  # put all electrodes (sensors) into geom
    geom.createNode(pos, marker=-99)  # just a historic convention
    geom.createNode(pos+pg.RVector3(0, -0.1))  # refine with 10cm

# pg.show(geom, boundaryMarker=1)
mesh = plc.createMesh(geom)
# pg.show(mesh)

scheme = pb.createData(elecs=np.linspace(-10, 10, 21),
                       schemeName='dd')
#        dumm,  1.S   2.S   3.S    Body
rhovec = np.array([0, 100.0, 50.0, 10.0, 100])  # ohm m
tauvec = np.array([0,   1e-3, 1e-3,  0.1,   1.0])  # s
mvec = np.array([0.001, 0.01, 0.001,  0.5,   0.5])  # - [0-1]
cvec = np.array([0.5,    0.5,  0.5,  0.5,   0.5])  # - [0-0.5]

frvec = [0.156, 0.312, 0.625, 1.25, 2.5, 5, 10, 20, 40, 80, 125,
         250, 500, 1000]  # SIP256C frequencies
sip = SIPdata(f=frvec, data=scheme)
sip.simulate(mesh, rhovec, tauvec, mvec, cvec)
print(sip)
sip.generateDataPDF(ipmax=100)
pg.wait()

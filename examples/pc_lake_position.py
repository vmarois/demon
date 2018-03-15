import vtk
import pandas as pd
import numpy as np
import os
import sys

examples_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
from depthmotionnet.vis import *


"""
** INITIAL TEST **
- Assume pointclouds have already been exported to CSV with export_pointcloud_to_csv()
- Assume pointclouds have dropped points towards infinity using find_depth_threshold() & ignore_points_depth_threshold() 
"""

# initial step: load back pointclouds from .csv files

# csv filenames: they are located in /examples for now TODO: dynamically go through the directories
scenes = ['thres_0382_pointcloud.csv', 'thres_0484_pointcloud.csv']

# Dummy translation vectors, TODO: extract the correct ones from image_auxiliary.csv
T = [np.array([100, 0, 0]), np.array([0, 100, 0])]

# Rotation matrices: Have to be constructed from theta-omega? TODO: extract the correct ones from image_auxiliary.csv
R = [np.eye(3), np.eye(3)]

# create renderer object
renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 0)
print 'Renderer created.\n'

# loop over filenames
for idx, fn in enumerate(scenes):
    # read .csv file
    scene = pd.read_csv(fn, index_col=0)
    print 'Imported CSV file.\n'

    # Extract points & colors from the pd.Dataframe
    points = scene[['X', 'Y', 'Z']].as_matrix()  # should return a np.array
    colors = scene[['R', 'G', 'B']].as_matrix()
    print 'Converted pd.Dataframes to np.array.\n'

    # apply first rotation to coordinates, then translation
    points = np.matmul(points,R[idx]) + T[idx]
    print "Applied dummy rotation & translation.\n"

    # next step would be to create a vtkActor from the pointcloud array
    scene_actor = create_pointcloud_actor(points=points, colors=colors)  # returns a vtk.vtkActor() object
    print "Created Scene Actor.\n"

    # add current scene actor to renderer
    renderer.AddActor(scene_actor)
    print "Added Scene Actor to renderer.\n"

    # create camera actor pointing at current scene
    camera_actor = create_camera_actor(R[idx], T[idx])
    print "Created Camera Actor.\n"
    renderer.AddActor(camera_actor)
    print "Added Camera Actor to renderer.\n"

# create 3D axes representation
axes = vtk.vtkAxesActor()
axes.GetXAxisCaptionActor2D().SetHeight(0.05)
axes.GetYAxisCaptionActor2D().SetHeight(0.05)
axes.GetZAxisCaptionActor2D().SetHeight(0.05)
axes.SetCylinderRadius(0.03)
axes.SetShaftTypeToCylinder()
# add axes to renderer
renderer.AddActor(axes)

# create renderer window
renwin = vtk.vtkRenderWindow()
renwin.SetWindowName("Point Cloud Lake Plan Viewer")
renwin.SetSize(800, 600)
renwin.AddRenderer(renderer)

# An interactor
interactor = vtk.vtkRenderWindowInteractor()
interstyle = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(interstyle)
interactor.SetRenderWindow(renwin)

# Start
interactor.Initialize()
interactor.Start()

import vtk
import pandas as pd
import numpy as np
from depthmotionnet.vis import *

"""
- Assume pointclouds have already been exported to CSV with export_pointcloud_to_csv()
- Assume pointclouds have dropped points towards infinity using find_depth_threshold() & ignore_points_depth_threshold() 
"""

# initial step: load back pointclouds from .csv files
scene_1 = pd.read_csv("/path/to/csv/scene_1.csv", index_col=0)
scene_2 = pd.read_csv("/path/to/csv/scene_2.csv", index_col=0)
scene_3 = pd.read_csv("/path/to/csv/scene_3.csv", index_col=0)

# next step: retrieve rotation & translation from image_auxiliary & apply to X,Y,Z coordinates

# first need to convert dataframe to np.ndarray
scene_1_coord = np.asmatrix(scene_1['X', 'Y', 'Z'].as_matrix())  # should return a np.array
scene_2_coord = np.asmatrix(scene_2['X', 'Y', 'Z'].as_matrix())
scene_3_coord = np.asmatrix(scene_3['X', 'Y', 'Z'].as_matrix())

# translation vectors
T1 = np.array(['x', 'y', 0])  # no translation on z-axis?
T2 = np.array(['x', 'y', 0])
T3 = np.array(['x', 'y', 0])

# rotation matrices: should be 3x3, constructed from theta-omega ?
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
R1 = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
R2 = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
R3 = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# apply first rotation to coordinates, then translation
scene_1_coord = scene_1_coord * R1 + T1
scene_2_coord = scene_2_coord * R2 + T2
scene_3_coord = scene_3_coord * R3 + T3

# next step would be to create a vtkActor from the pointcloud array
scene1_actor = create_pointcloud_actor(scene_1_coord)  # returns a vtk.vtkActor() object
scene2_actor = create_pointcloud_actor(scene_2_coord)
scene3_actor = create_pointcloud_actor(scene_3_coord)

# create renderer object
renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 0)

# add actors to renderer
renderer.AddActor(scene1_actor)
renderer.AddActor(scene2_actor)
renderer.AddActor(scene3_actor)

# have no idea what this is
axes = vtk.vtkAxesActor()
axes.GetXAxisCaptionActor2D().SetHeight(0.05)
axes.GetYAxisCaptionActor2D().SetHeight(0.05)
axes.GetZAxisCaptionActor2D().SetHeight(0.05)
axes.SetCylinderRadius(0.03)
axes.SetShaftTypeToCylinder()
renderer.AddActor(axes)

# create renderer window
renwin = vtk.vtkRenderWindow()
renwin.SetWindowName("Point Cloud Viewer")
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

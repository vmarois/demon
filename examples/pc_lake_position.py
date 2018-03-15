import vtk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

examples_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
from depthmotionnet.vis import *


"""
** INITIAL TEST **
- Assume pointclouds have already been exported to CSV with export_pointcloud_to_csv()
"""

# csv filenames (entire pointcloud): they are located in /examples for now TODO: dynamically go through the directories
scenes = ['0382_pointcloud.csv', '0484_pointcloud.csv']

# set the path to the dataset TODO: Retrieve the directory from which we took the example '0382.jpg' etc
DATA_PATH = 'cs-share/pradalier/lake/Dataset/2016/'

#taking image_auxilliary from 'Dataset/2016/160620' as an example for now TODO: How to browse in absolute path?
image_auxilliary = pd.read_csv('image_auxilliary.csv')
print "Correctly loaded image_auxilliary as pd.Dataframe"
print image_auxilliary.head()
"""
TODO next:
Construct the filepath using 'seq' to get access to the source images (depends on the directory structure)
For a given image, extract x, y, theta, pan, tilt to construct translation vector & rotation matrix
Apply these to the pointcloud and check display
"""
# Dummy translation vectors, TODO: extract the correct ones from image_auxiliary.csv
T = [np.array([10, 0, 0]), np.array([0, 10, 0])]

# Rotation matrices: Have to be constructed from theta-omega? TODO: extract the correct ones from image_auxiliary.csv
R = [np.eye(3), np.eye(3)]

# create renderer object
renderer = vtk.vtkRenderer()
renderer.SetBackground(0, 0, 0)
print 'Renderer created.\n'

# loop over filenames
for idx, fn in enumerate(scenes):
    # read .csv file containing the entire pointcloud
    scene = pd.read_csv(fn, index_col=0)
    print 'Imported CSV file.\n'

    # find the depth threshold
    n, bins, patches = plt.hist(scene['Z'], bins=40, normed=True)  # extract info from the histogram
    threshold = np.ceil(bins[np.argmax(n) + 1])  # select the bin directly on the right of the argmax bin
    print '\nDepth threshold : ', threshold

    # keep only points with Z < threshold
    scene = scene.ix[scene['Z'] <= threshold]
    print "Removed points with Z > depth threshold"

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

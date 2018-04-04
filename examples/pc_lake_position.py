import vtk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from itertools import tee
from scipy.spatial import distance

examples_dir = os.path.dirname(__file__)
pointclouds_dir = os.path.join(examples_dir, "pointclouds/")
pointclouds_thres_dir = os.path.join(examples_dir, "pointclouds_thres/")

image_auxilliary = "160808/image_auxilliary_new.csv"

sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
from depthmotionnet.vis import *


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def readTupleList(filename):
    """
    read a file containing image pair reference to a list
    one line should contain paths "img1.name img2.name"
    """
    list = []
    for line in open(filename).readlines():
        if line.strip() != '':
            list.append(line.split())

    return list


def read_csv_values(file, image_ref):
    """
    Parse image_auxiliary.csv to retrieve [x, y, theta, pan, tilt, omega] for the specified image reference
    :param file: path to image_auxiliary.csv
    :param image_ref: corresponding 'seq' key for the image
    :return: [x, y, theta, pan, tilt, omega] as a np.array
    """
    file = open(file)
    dataframe = pd.read_csv(file)
    dataframe = np.array(dataframe.values)
    image_list = dataframe[:, 1]
    index = (np.abs(image_list - image_ref)).argmin()
    x = dataframe[index, 2]
    y = dataframe[index, 3]
    theta = dataframe[index, 4]
    pan = dataframe[index, 5]
    tilt = dataframe[index, 6]
    omega = dataframe[index, 14]
    data = np.array([x, y, theta, pan, tilt, omega])

    return data


def rotation_matrix(theta, pan, tilt):
    """
    Compute the rotation matrix to apply to the pointcloud.
    (theta-pan) around Z axis
    (-tilt) around Y axis
    Using elementary rotation matrix formula for rotation about an axis.
    :return: global 3x3 rotation matrix
    """
    Ry = np.matrix([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(-tilt)]])
    Rz = np.matrix([[np.cos(theta-pan), -np.sin(theta-pan), 0], [np.sin(theta-pan), np.cos(theta-pan), 0], [0, 0, 1]])
    return np.array(Ry * Rz)


def compute_center_scale(image_auxilliary):
    """
    Parse image_auxilliary to compute the average (x, y) coordinates.
    Also computes the 'scale factor' (euclidean norm between the 2 images location) for each scene.
    Stores them in file for future reuse.
    :param image_auxilliary: filepath to image_auxilliary.csv corresponding to the dataset.
    :return: x_mean, y_mean
    """
    # start by building a list of the 'seq' values corresponding to the .csv in pointclouds/
    seq_index = [f[:5] for f in os.listdir(pointclouds_dir) if os.path.isfile(os.path.join(pointclouds_dir, f))]
    # sort the list by increasing order: important to compute the 'scale factor' between 2 successives frames
    seq_index.sort(key=float)

    # for the translation, we need to get x_mean, y_mean first (translation vector = (x-x_mean, y-y_mean, 0))
    x_mean = 0
    y_mean = 0
    coord = []

    # loop over the 'seq' key values
    for seq in seq_index:
        # parse image_auxilliary for the specified key value
        param = read_csv_values(image_auxilliary, float(seq))
        x_mean += param[0]
        y_mean += param[1]
        # append x,y coordinates to compute the euclidean distance
        coord.append((param[0], param[1]))

    x_mean /= len(seq_index)
    y_mean /= len(seq_index)

    print "Computed x_mean =", x_mean, ",y_mean =", y_mean, "\n"

    # write to file
    with open('center_coord.txt', 'w') as fp:
        fp.write('%s %s' % (x_mean, y_mean))
    print "Saved center coordinates to file."

    distances = []
    for elt in pairwise(coord):  # elt is a tuple (x, y)
        distances.append(distance.euclidean(elt[0], elt[1]))

    # write to file
    with open('distances.txt', 'w') as fp:
        fp.write('\n'.join('%s' % x for x in distances))
    print "Saved distances to file."

    return x_mean, y_mean


"""
** INITIAL TEST **
- Assumes pointclouds have already been exported to CSV with export_pointcloud_to_csv(), and that they are located in
examples/pointclouds/
- Assumes the image_auxilliary file used is in 160808/
"""


def visualization(n_pointclouds, image_auxilliary):
    """
    Create rendered window to visualize n_pointclouds placed on the lake plan.
    :param n_pointclouds: number of pointclouds to visualize
    :param image_auxilliary: filepath of image_auxilliary
    :return: start the renderer window
    """
    # start by building a list of the 'seq' values corresponding to the .csv in pointclouds/
    seq_index = [f[:5] for f in os.listdir(pointclouds_dir) if os.path.isfile(os.path.join(pointclouds_dir, f))]
    # sort the list by increasing order (to respect order of frames -> to draw the overall trajectory)
    seq_index.sort(key=float)

    # create renderer object
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    print 'Renderer created.\n'

    # check if the average (x, y) coordinates have been already been computed & saved to file, else compute them
    if os.path.isfile('center_coord.txt'):
        coord = readTupleList('center_coord.txt')[0]
        x_mean, y_mean = [float(c) for c in coord]
    else:
        x_mean, y_mean = compute_center_scale(image_auxilliary)

    # read from files the distances between the image positions to scale the pointclouds
    # the 'distances.txt' file should exist as it is created by compute_center_scale()
    with open('distances.txt') as f:
        distances = f.read().splitlines()
    distances = [float(i) for i in distances]

    # only display n_pointclouds in total
    interval = len(seq_index) // n_pointclouds  # floor division
    print 'Only displaying a maximum of {} pointclouds.'.format(n_pointclouds)

    # draw the trajectory as a sequence of lines between vtkPoints
    # create a vtkPoints object to store the points
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(len(seq_index))

    # create a cell array to store the lines
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(seq_index)+1)

    # loop over 'seq' key values
    for idx, seq in enumerate(seq_index):

        # parse image_auxilliary to get x, y to draw a line
        param = read_csv_values(image_auxilliary, float(seq))  # [x, y, theta, pan, tilt, omega]

        # store point corresponding to current poincloud position
        pts.SetPoint(idx, param[0]-x_mean, param[1]-y_mean, 0.0)
        lines.InsertCellPoint(idx)

        # only display n_pointclouds max.
        if idx % interval != 0:
            continue

        # read .csv file containing the entire pointcloud
        filename = pointclouds_dir + seq + '_pc.csv'
        scene = pd.read_csv(filename, index_col=0)
        print 'Imported CSV file.\n'

        # multiply coordinates by the distance between the 2 images to have an idea of the scale
        scene[['X', 'Y', 'Z']] *= distances[idx]

        # keep only points with Z < 20 (arbitrary limit)
        scene = scene.ix[scene['Z'] <= 20]
        print "Removed points with Z > 20"

        # Extract points & colors from the pd.Dataframe
        points = scene[['X', 'Y', 'Z']].as_matrix()  # should return a np.array
        colors = scene[['R', 'G', 'B']].as_matrix()
        print 'Converted pd.Dataframes to np.array.\n'

        # invert coordinates to go from camera viewpoint to world coordinate system: rotation of PI/2 about x-axis ?
        X_coord = np.copy(points[:, 0])
        Y_coord = np.copy(points[:, 1])
        points[:, 0] = points[:, 2]  # x <- z
        points[:, 1] = - X_coord  # y <- -x
        points[:, 2] = - Y_coord  # z <- -y

        # construct rotation matrix
        rot_matrix = rotation_matrix(param[2], param[3], param[4])

        # apply first rotation to coordinates, then translation
        points = np.matmul(points, rot_matrix)
        points += np.array([param[0] - x_mean, param[1] - y_mean, 0])  # we need to add an offset on the z-coordinate
        print "Applied rotation & translation.\n"

        # create a vtkActor from the pointcloud array
        scene_actor = create_pointcloud_actor(points=points, colors=colors)  # returns a vtk.vtkActor() object
        print "Created Scene Actor.\n"

        # add current scene actor to renderer
        renderer.AddActor(scene_actor)
        print "Added Scene Actor to renderer.\n"

        # create rotation matrix for camera actor
        cam_rot_matrix = np.matmul(np.array([[0,-1, 0], [0, 0, -1], [1, 0, 0]]), rot_matrix)
        # create a vtkActor to represent the camera point of view
        camera_actor = create_camera_actor(cam_rot_matrix, np.zeros((3,)))

        # set camera actor position
        camera_actor.SetPosition(param[0] - x_mean, param[1] - y_mean, 0)
        print "Created Camera Actor.\n"

        # add current camera actor to renderer
        renderer.AddActor(camera_actor)
        print "Added Camera Actor to renderer.\n"

    lines.InsertCellPoint(0)

    # vtkPolyData represents a geometric structure consisting of lines
    polygon = vtk.vtkPolyData()
    polygon.SetPoints(pts)
    polygon.SetLines(lines)

    # vtkPolyDataMapper is a class that maps polygonal data to graphics primitives
    polygonMapper = vtk.vtkPolyDataMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        polygonMapper.SetInputConnection(polygon.GetProducerPort())
    else:
        polygonMapper.SetInputData(polygon)
        polygonMapper.Update()

    polygonActor = vtk.vtkActor()
    polygonActor.SetMapper(polygonMapper)
    renderer.AddActor(polygonActor)

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


if __name__ == '__main__':
    #x_mean, y_mean = compute_center_scale(image_auxilliary)
    visualization(n_pointclouds=50, image_auxilliary=image_auxilliary)

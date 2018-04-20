import vtk
import pandas as pd
import numpy as np
import os
import sys
from itertools import tee

examples_dir = os.path.dirname(__file__)

#### PATHS TO CHANGE WHEN NEEEDED ####
pointclouds_dir = os.path.join(examples_dir, "/mnt/dataX/pointclouds/")
image_auxilliary = "/mnt/dataX/160808f/image_auxilliary.csv"
center_coord = "output/center_coord.txt"
distances = 'output/distances_baseline.txt'
######################################

sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
from depthmotionnet.vis import *


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
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
    DEPRECATED: Too slow, as it loads the file back from disk everytime + doesn't work on full dataset?
        Use fetch_values() instead.
    Parse image_auxiliary.csv to retrieve [x, y, theta, pan, tilt, omega] for the specified image reference.

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


def fetch_values(array, image_seq):
    """
    Returns [x, y, theta, pan, tilt, omega] for the specified image reference

    :param array: np.array created from image_auxilliary (loading it from memory)
    :param image_seq: seq key for the current image

    :return: [x, y, theta, pan, tilt, omega]
    """
    data = array[array[:, 1] == image_seq][0][[2, 3, 4, 5, 6, 14]]
    return data


def rotation_matrix(theta, pan, tilt):
    """
    Compute the rotation matrix to apply to the pointcloud:
        - (theta-pan) around Z axis
        - (tilt) around Y axis
    Using elementary rotation matrix formula for rotation about an axis.

    :param theta, pan, tilt: rotation angles.

    :return: global 3x3 rotation matrix
    """
    Ry = np.matrix([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    Rz = np.matrix([[np.cos(theta-pan), -np.sin(theta-pan), 0], [np.sin(theta-pan), np.cos(theta-pan), 0], [0, 0, 1]])

    return np.array(Ry * Rz)


"""
** IMPORTANT REMARKS **
- Assumes pointclouds have already been exported to CSV with export_pointcloud_to_csv(), and that they are located in
pointclouds_dir
- Assumes the image_auxilliary file exists, will refer to filepath image_auxilliary to search for it.
- Assumes the average x,y coordinates have been already computed, will refer to filepath center_coord to search for it.
- Assumes the distances between 2 images of each pair have been already computed, will refer to filepath distances.
"""


def visualization(n_pointclouds, image_auxilliary, center_coord, distances, z_limit):
    """
    Create rendered window to visualize n_pointclouds placed on the lake plan.

    :param n_pointclouds: number of pointclouds to visualize
    :param image_auxilliary: filepath of image_auxilliary
    :param center_coord: filepath where are stored average x,y coordinates.
    :param distances: filepath where are stored distances between the 2 images of each pair.
    :param z_limit: int, depth limit, will remove the points with z coordinate > z_limit

    :return: start the renderer window
    """
    # start by building a list of the 'seq' values corresponding to the .csv in pointclouds_dir/
    seq_index = [f[:5] for f in os.listdir(pointclouds_dir) if os.path.isfile(os.path.join(pointclouds_dir, f))]
    # sort the list by increasing order (to respect order of frames -> to draw the overall trajectory)
    seq_index.sort(key=float)

    renderer = vtk.vtkRenderer()  # create renderer object
    renderer.SetBackground(0, 0, 0)
    print 'Renderer created.\n'

    # check if the average (x, y) coordinates have been already been computed & saved to file, else raise error
    try:
        coord = readTupleList(center_coord)[0]
        x_mean, y_mean = [float(c) for c in coord]
    except IOError:
        print "Could not load x_mean, y_mean from file, please compute them using image_browser.py."

    # read from files the distances between the image positions to scale the pointclouds
    # the 'distances.txt' file should exist as it is created by compute_center_scale()
    try:
        with open(distances) as f:
            dist = f.read().splitlines()
        dist = [float(i) for i in dist]
    except IOError:
        print "Could not load 'distances.txt' from file, please compute them using image_browser.py."

    # only display n_pointclouds in total
    interval = len(seq_index) // n_pointclouds  # floor division
    print 'Only displaying a maximum of {} pointclouds.'.format(n_pointclouds)

    # draw the trajectory as a sequence of lines between vtkPoints
    pts = vtk.vtkPoints()  # create a vtkPoints object to store the points
    pts.SetNumberOfPoints(len(seq_index))

    lines = vtk.vtkCellArray()  # create a cell array to store the lines
    lines.InsertNextCell(len(seq_index)+1)

    # parse csv file into np.array
    img_aux = [[float(s) for s in l.strip().split(",")] for l in open(image_auxilliary, 'r').readlines() if l[0] != '%']
    img_aux = np.asarray(img_aux)

    # there can be a mismatch between the number of images in dataset & number of lines in image_auxilliary:
    min_seq = int(img_aux[0, 1])
    max_seq = int(img_aux[-1, 1])

    # loop over 'seq' key values
    for idx, seq in enumerate(seq_index):

        if int(seq) in range(min_seq, max_seq + 1):  # check if entry exists in image_auxilliary

            param = fetch_values(img_aux, int(seq))  # get [x, y, theta, pan, tilt, omega] for first image of the pair

            # store point corresponding to current poincloud position
            pts.SetPoint(idx, param[0]-x_mean, param[1]-y_mean, 0.0)
            lines.InsertCellPoint(idx)

            if idx % interval != 0:  # only display n_pointclouds maximum
                continue

            # read .csv file containing the entire pointcloud
            filename = pointclouds_dir + str(seq) + '_pc.csv'
            scene = pd.read_csv(filename, index_col=0)
            print 'Imported CSV file.\n'

            # multiply coordinates by the distance between the 2 images to have an idea of the scale
            scene[['X', 'Y', 'Z']] *= dist[idx]

            # keep only points with Z < 20 (arbitrary limit)
            scene = scene.ix[scene['Z'] <= z_limit]
            print "Removed points with Z > {}".format(z_limit)

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

            # construct rotation matrix from tilt, theta, pan
            rot_matrix = rotation_matrix(param[2], param[3], param[4])

            # apply first rotation to coordinates, then translation
            points = np.matmul(points, rot_matrix)
            points += np.array([param[0] - x_mean, param[1] - y_mean, 0])
            print "Applied rotation & translation.\n"

            # create a vtkActor from the pointcloud array
            scene_actor = create_pointcloud_actor(points=points, colors=colors)  # returns a vtk.vtkActor() object
            print "Created Scene Actor.\n"

            # add current scene actor to renderer
            renderer.AddActor(scene_actor)
            print "Added Scene Actor to renderer.\n"

            # create rotation matrix for camera actor
            cam_rot_matrix = np.matmul(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]), rot_matrix)
            # create a vtkActor to represent the camera point of view with created rotation matrix
            camera_actor = create_camera_actor(cam_rot_matrix, np.zeros((3,)))

            # set camera actor position
            camera_actor.SetPosition(param[0] - x_mean, param[1] - y_mean, 0)
            print "Created Camera Actor.\n"

            # add current camera actor to renderer
            renderer.AddActor(camera_actor)
            print "Added Camera Actor to renderer.\n"

        else:
            print "Image with seq: {} not in image_auxilliary. Can't find parameters values.".format(seq)

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
    visualization(n_pointclouds=30,
                  image_auxilliary=image_auxilliary,
                  center_coord=center_coord,
                  distances=distances,
                  z_limit=20)

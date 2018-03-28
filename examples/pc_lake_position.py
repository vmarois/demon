import vtk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

examples_dir = os.path.dirname(__file__)
pointclouds_dir = os.path.join(examples_dir, "pointclouds/")
pointclouds_thres_dir = os.path.join(examples_dir, "pointclouds_thres/")

image_auxilliary = "160808/image_auxilliary_new.csv"

sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
from depthmotionnet.vis import *


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
    First need to rotate the pointcloud M_PI/2 along the x-axis then apply:
    (theta-pan) around Z axis
    (-tilt) around Y axis
    :return: global 3x3 rotation matrix
    """
    Rx = np.matrix([[1, 0, 0], [0, np.cos(-np.pi/2), -np.sin(-np.pi/2)], [0, np.sin(-np.pi/2), np.cos(-np.pi/2)]])
    Ry = np.matrix([[np.cos(-tilt), 0, np.sin(-tilt)], [0, 1, 0], [-np.sin(-tilt), 0, np.cos(-tilt)]])
    Rz = np.matrix([[np.cos(theta-pan), -np.sin(theta-pan), 0], [np.sin(theta-pan), np.cos(theta-pan), 0], [0, 0, 1]])
    return np.array(Rx * Ry * Rz)


def compute_center(image_auxilliary):
    """
    Parse image_auxilliary to compute the average (x, y) coordinates.
    Stores them in file for future reuse.
    :param image_auxilliary: filepath to image_auxilliary.csv corresponding to the dataset.
    :return:
    """
    # start by building a list of the 'seq' values corresponding to the .csv in pointclouds/
    seq_index = [f[:5] for f in os.listdir(pointclouds_dir) if os.path.isfile(os.path.join(pointclouds_dir, f))]

    # for the translation, we need to get x_mean, y_mean first (translation vector = (x-x_mean, y-y_mean, 0))
    x_mean = 0
    y_mean = 0

    for seq in seq_index:
        param = read_csv_values(image_auxilliary, float(seq))
        x_mean += param[0]
        y_mean += param[1]

    x_mean /= len(seq_index)
    y_mean /= len(seq_index)
    print "Computed x_mean =", x_mean, ",y_mean =", y_mean, "\n"

    # write to file
    with open(os.path.join(pointclouds_dir, 'center_coord.txt'), 'w') as fp:
        fp.write('%s %s' % (x_mean, y_mean))

    print "Saved center coordinates to file."
    return x_mean, y_mean


"""
** INITIAL TEST **
- Assume pointclouds have already been exported to CSV with export_pointcloud_to_csv(), and that they are located in
examples/pointclouds/
- Assume the image_auxilliary file used is in 160808/
"""


def visualization(n_pointclouds, image_auxilliary):
    """
    Create rendered window to visualize n_pointclouds placed on the lake plan.
    :param n_pointclouds: number of pointclouds to visualize
    """
    # start by building a list of the 'seq' values corresponding to the .csv in pointclouds/
    seq_index = [f[:5] for f in os.listdir(pointclouds_dir) if os.path.isfile(os.path.join(pointclouds_dir, f))]

    # create renderer object
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0, 0, 0)
    print 'Renderer created.\n'

    # check if average (x, y) coordinates have been already been computed & saved to file, else compute them
    if os.path.isfile(os.path.join(pointclouds_dir, 'center_coord.txt')):
        coord = readTupleList(os.path.join(pointclouds_dir, 'center_coord.txt'))[0]
        x_mean, y_mean = [float(c) for c in coord]
    else:
        x_mean, y_mean = compute_center(image_auxilliary)

    # only display n_pointclouds in total
    interval = len(seq_index) // n_pointclouds
    print 'Only displaying a maximum of {} pointclouds.'.format(n_pointclouds)

    count = 0
    # loop over filenames
    for seq in seq_index:
        count += 1
        if count % interval != 0:
            continue

        # read .csv file containing the entire pointcloud
        filename = pointclouds_dir + seq + '_pc.csv'
        scene = pd.read_csv(filename, index_col=0)
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

        # parse image_auxilliary to get x, y, theta, pan, tilt, omega
        param = read_csv_values(image_auxilliary, float(seq))

        # construct rotation matrix
        rot_matrix = rotation_matrix(param[2], param[3], param[4])
        # apply first rotation to coordinates, then translation
        points = np.matmul(points, rot_matrix) + np.array([param[0] - x_mean, param[1] - y_mean, 0])
        print "Applied rotation & translation.\n"

        # next step would be to create a vtkActor from the pointcloud array
        scene_actor = create_pointcloud_actor(points=points, colors=colors)  # returns a vtk.vtkActor() object
        print "Created Scene Actor.\n"

        # add current scene actor to renderer
        renderer.AddActor(scene_actor)
        print "Added Scene Actor to renderer.\n"

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
    #x_mean, y_mean = compute_center(image_auxilliary)
    visualization(n_pointclouds=15, image_auxilliary=image_auxilliary)

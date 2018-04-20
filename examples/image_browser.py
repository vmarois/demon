"""
Go through the dataset folder and group images by pairs to pass them to DeMoN in one forward pass.
"""
import os
import numpy as np
import pandas as pd

examples_dir = os.path.dirname(__file__)

###### PARAMS TO CHANGE WHEN NEEDED ######
dataset_dir = os.path.join('/', "/mnt/dataX/160808f/")
image_auxilliary = "/mnt/dataX/160808f/image_auxilliary.csv"
##########################################


def group_images_pairs():
    """
    Group images by simple pairs without any selection.
    From a list [img1, img2, img3, img4 ... imgn], it creates pairs (img1, img2), (img3, img4) etc.
    Create a .txt file containing the filepath to the 2 images of 1 pair on the same line:
    /path/to/img1   /path/to/img2
    /path/to/img3   /path/to/img4

    Also computes the distances between the 2 images for each pair (using the image_auxilliary) and saves it to file.
    Will be used later as a scale factor for visualization.

    Also computes average x,y coordinates and saves them to file (used for visualization)
    """
    images_names = []
    seq = []

    # walk through dataset dir to collect filepaths & construct seq keys
    for path, subdirs, _ in os.walk(dataset_dir):
        for dir in sorted(subdirs):
            for f in sorted(os.listdir(os.path.join(path, dir))):
                if os.path.isfile(os.path.join(path, dir, f)):
                    filename = os.path.join(path, dir, f)
                    images_names.append(filename)
                    seq.append(int(filename[-11:-9] + filename[-7:-4]))

    # group images names & seq by pairs
    images_pairs = zip(*[images_names[i::2] for i in range(2)])
    seq_pairs = zip(*[seq[i::2] for i in range(2)])

    # write image pairs to file
    with open('output/image_pairs.txt', 'w') as fp:
        fp.write('\n'.join('%s %s' % x for x in images_pairs))
    print "File output/image_pairs.txt created."

    # parse csv file into np.array
    img_aux = [[float(s) for s in l.strip().split(",")] for l in open(image_auxilliary, 'r').readlines() if l[0] != '%']
    img_aux = np.asarray(img_aux)

    # compute x_mean, y_mean:
    x_mean = np.mean(img_aux[:, 2])
    y_mean = np.mean(img_aux[:, 3])

    # write center coordinates to file
    with open('output/center_coord.txt', 'w') as fp:
        fp.write('%s %s' % (x_mean, y_mean))
    print "Saved center coordinates to file."

    # there can be a mismatch between the number of images in dataset & number of lines in image_auxilliary:
    min_seq = int(img_aux[0,1])
    max_seq = int(img_aux[-1,1])

    # compute associated distance (will be used as scale factor for pc_lake_position)
    distances = []
    for pair in seq_pairs:
        if pair[0] in range(min_seq, max_seq+1):
            img1 = img_aux[img_aux[:,1] == pair[0]][0]
            img2 = img_aux[img_aux[:, 1] == pair[1]][0]
            distances.append(np.sqrt( (img1[2]-img2[2]) ** 2 + (img1[3]-img2[3]) ** 2 ))
        else:
            continue

    # write distances to file
    with open('output/distances.txt', 'w') as fp:
        fp.write('\n'.join('%s' % x for x in distances))
    print "File output/distances.txt created."


def read_csv_values(file, image_ref):
    """
    Parse image_auxiliary.csv to retrieve [x, y, theta, pan, tilt, omega] for the specified image reference
    :param file: path to image_auxiliary.csv
    :param image_ref: corresponding 'seq' key for the image
    :return: [x, y, theta, pan, tilt, omega] as a np.array
    """
    file = open(file)
    dataframe = pd.read_csv(file)  # the .csv is badly formatted, numeric values are passed as str
    dataframe = dataframe[dataframe['seq'] != 'seq']  # remove lines containing keys as values

    # convert dtype to float
    dataframe = dataframe.apply(pd.to_numeric)

    min_seq = int(dataframe['seq'].iloc[0])
    max_seq = int(dataframe['seq'].iloc[-1])

    # ensure seq key value is in appropriate range
    if int(image_ref) in range(min_seq, max_seq+1):
        # extract useful values
        x, y, theta, pan, tilt, omega = dataframe[dataframe['seq'] == image_ref].iloc[0][['x', 'y', 'theta', 'pan', 'tilt',
                                                                                      'omega']]
        data = [x, y, theta, pan, tilt, omega]
        return data
    else:
        print "Seq key value not in image_auxiliary range, returning None"


def group_images_baseline(dataset_dir, image_auxilliary):
    """
    Attempt to group the images by pairs based on several criteria:
        - the distance between their 2 positions: Try to get a distance close to 1m. (0.97 <= dist <= 1.03)
        - The tilt angle difference: should be small enough (i.e. to avoid frames of camera pointing to sky)
        - The (theta-pan) angle difference: should also be small, to select frames of the same scene ?
    Also saves the computed distance to file (used a scale factor for visualization)
    :return: write (img1_name, img2_name) + distances to .txt files
    """
    # get image names: e.g. '160808/0039/0671.jpg' and construct the 'seq' key from here:
    imgnames_seq = []

    for path, subdirs, _ in os.walk(dataset_dir):
        for dir in sorted(subdirs):
            for f in sorted(os.listdir(os.path.join(path, dir))):
                if os.path.isfile(os.path.join(path, dir, f)):
                    img_name = os.path.join(path, dir, f)
                    seq = int(img_name[-11:-9] + img_name[-7:-4])
                    imgnames_seq.append((img_name, seq))

    # there can be a mismatch between the number of images in dataset & number of lines in image_auxilliary:
    min_seq = int(imgnames_seq[0][1])
    max_seq = int(imgnames_seq[-1][1])

    # parse csv files np.array
    img_aux = [[float(s) for s in l.strip().split(",")] for l in open(image_auxilliary, 'r').readlines() if l[0] != '%']
    img_aux = np.asarray(img_aux)

    # compute x_mean, y_mean:
    x_mean = np.mean(img_aux[:, 2])
    y_mean = np.mean(img_aux[:, 3])

    # write center coordinates to file
    with open('output/center_coord.txt', 'w') as fp:
        fp.write('%s %s' % (x_mean, y_mean))
    print "Saved center coordinates to file."

    image_pair = []
    distances = []
    # loop over the dataset, to select 2 pictures respecting conditions stated in docstring
    idx = 0

    while idx < len(img_aux)-2:
        # get corresponding line from parsed image_auxilliary
        img_r = img_aux[idx]

        # check that image exists in dataset folder
        if img_r[1] in range(min_seq, max_seq+1):
            print "Image:", img_r[1]

            # define new index from the current, is used to loop from the current frame to find match
            idx_2 = idx
            idx += 1

            while idx_2 < len(img_aux)-2:

                idx_2 += 1

                # read line corresponding to second image
                img_l = img_aux[idx_2]

                # calculate distance, angle difference between the 2 frames
                dist = np.sqrt((img_r[2] - img_l[2]) ** 2 + (img_r[3] - img_l[3]) ** 2)
                diff_tilt = np.abs(img_r[6] - img_l[6])
                diff_theta_pan = np.abs((img_r[4] - img_r[5]) - (img_l[4] - img_l[5]))

                # if conditions are checked,
                if (0.97 <= dist <= 1.03) & (diff_tilt < 0.01) & (diff_theta_pan < 0.3):
                    print "found image:", img_l[1], "distance:", dist
                    distances.append(dist)

                    # get images filepath
                    img1 = [item[0] for item in imgnames_seq if item[1] == img_r[1]]
                    img2 = [item[0] for item in imgnames_seq if item[1] == img_l[1]]
                    image_pair.append((img1[0], img2[0]))
                    idx = idx_2
                    idx_2 = 0
                    break

    # save matched image pairs to file
    with open('output/image_pairs_baseline.txt', 'w') as fp:
        fp.write('\n'.join('{} {}'.format(x[0], x[1]) for x in image_pair))
    print "File output/image_pairs_baseline.txt created."

    # write distances to file
    with open('output/distances_baseline.txt', 'w') as fp:
        fp.write('\n'.join('%s' % x for x in distances))
    print "File output/distances_baseline.txt created."


if __name__ == '__main__':
    #group_images_pairs()

    group_images_baseline(dataset_dir, image_auxilliary)

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import sys

# define dir paths
examples_dir = os.path.dirname(__file__)
pointclouds_dir = os.path.join(examples_dir, "pointclouds/")
pointclouds_thres_dir = os.path.join(examples_dir, "pointclouds_thres/")
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

from depthmotionnet.networks_original import *
from depthmotionnet.vis import *


class DeMoN:

    def __init__(self):
        """
        init the network & load back weights
        """
        # check gpu availability & set data formats
        if tf.test.is_gpu_available(True):
            self.data_format = 'channels_first'
        else:  # running on cpu requires channels_last data format
            self.data_format = 'channels_last'

        # set gpu options
        self.gpu_options = tf.GPUOptions()
        self.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=self.gpu_options))

        # init networks
        self.bootstrap_net = BootstrapNet(self.session, self.data_format)
        self.iterative_net = IterativeNet(self.session, self.data_format)
        self.refine_net = RefinementNet(self.session, self.data_format)

        self.session.run(tf.global_variables_initializer())

        # load weights
        self.saver = tf.train.Saver()
        self.saver.restore(self.session, os.path.join(weights_dir, 'demon_original'))

    def run_many(self, filename):
        """
        Wrap self.run() into a loop to handle a list of image pairs.
        """
        # parse filename containing img pairs references
        img_pairs_list = self.readTupleList(filename)

        for pairs in img_pairs_list:
            img1_name = pairs[0]
            img2_name = pairs[1]
            self.run(img1_name, img2_name)

    def run(self, img1_name, img2_name):
        """
        run the network & get predictions for a single image pair.
        """
        img1 = Image.open(os.path.join(examples_dir, img1_name))
        img2 = Image.open(os.path.join(examples_dir, img2_name))
        input_data = self.prepare_input_data(img1, img2)

        # run the BootstrapNet network
        result = self.bootstrap_net.eval(input_data['image_pair'], input_data['image2_2'])

        # run the IterativeNet 3 times
        for i in range(3):
            result = self.iterative_net.eval(
                input_data['image_pair'],
                input_data['image2_2'],
                result['predict_depth2'],
                result['predict_normal2'],
                result['predict_rotation'],
                result['predict_translation']
            )

        # get motion estimation
        rotation = result['predict_rotation']
        translation = result['predict_translation']

        # run the RefinementNet
        result = self.refine_net.eval(input_data['image1'], result['predict_depth2'])

        # construct the output file path to export full pointcloud to .csv
        full_pc_name = pointclouds_dir + img1_name[-11:-9] + img1_name[-7:-4] + '_pc.csv'

        # try to export the full pointcloud to .csv
        try:
            export_pointcloud_to_csv(
                filename=full_pc_name, inverse_depth=result['predict_depth0'],
                image=input_data['image_pair'][0, 0:3] if self.data_format == 'channels_first' else
                input_data['image_pair'].transpose([0, 3, 1, 2])[0, 0:3],
                rotation=rotation, translation=translation)

        except ImportError as err:
            print("Cannot export to csv :", err)
        print '\nPointcloud saved to .csv file.'

        # try to find the depth threshold
        threshold = find_depth_threshold(full_pc_name)
        print '\nDepth threshold : ', threshold

        # construct the output file path to export thresholded pointcloud to .csv
        thres_pc_name = pointclouds_thres_dir + img1_name[-11:-9] + img1_name[-7:-4] + '_pc_thres.csv'

        # Ignore points located at infinity (on the sky plane) in the pointcloud
        ignore_points_depth_threshold(full_pc_name, threshold, thres_pc_name)
        print "CSV file has been processed to remove points with Z > threshold."

    def prepare_input_data(self, img1, img2):
        """
        Creates the arrays used as input from the two images.
        """
        # scale images if necessary
        if img1.size[0] != 256 or img1.size[1] != 192:
            img1 = img1.resize((256, 192))
        if img2.size[0] != 256 or img2.size[1] != 192:
            img2 = img2.resize((256, 192))
        img2_2 = img2.resize((64, 48))

        # transform range from [0,255] to [-0.5,0.5]
        img1_arr = np.array(img1).astype(np.float32) / 255 - 0.5
        img2_arr = np.array(img2).astype(np.float32) / 255 - 0.5
        img2_2_arr = np.array(img2_2).astype(np.float32) / 255 - 0.5

        if self.data_format == 'channels_first':
            img1_arr = img1_arr.transpose([2, 0, 1])
            img2_arr = img2_arr.transpose([2, 0, 1])
            img2_2_arr = img2_2_arr.transpose([2, 0, 1])
            image_pair = np.concatenate((img1_arr, img2_arr), axis=0)
        else:
            image_pair = np.concatenate((img1_arr, img2_arr), axis=-1)

        img_pair = {
            'image_pair': image_pair[np.newaxis, :],
            'image1': img1_arr[np.newaxis, :],  # first image
            'image2_2': img2_2_arr[np.newaxis, :],  # second image with (w=64,h=48)
        }
        return img_pair

    def readTupleList(self, filename):
        """
        read a file containing image pair reference to a list
        one line should contain paths "img1.name img2.name"
        """
        list = []
        for line in open(filename).readlines():
            if line.strip() != '':
                list.append(line.split())

        return list


if __name__ == '__main__':
        demon = DeMoN()
        #demon.run_many('image_pairs_man.txt')
        demon.run_many('image_pairs.txt')
        print "Success"

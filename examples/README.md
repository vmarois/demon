# MAROIS Vincent - Special Problem Code Documentation

_This README file gathers information on how the code is structured and how to use it._

Main environment: `Python 2.7`

`Python` packages requirements:
* Tensorflow 1.4
* VTK
* Numpy
* Pillow
* Pandas
* (Matplotlib)

Main Github repo: https://github.com/vmarois/demon

    git clone https://github.com/vmarois/demon.git


The code written is located in `demon/examples` and some functions are located in `demon/python/depthmotionnet/vis.py`.

The code is using the partition `/mnt/dataX` on GPU3 to store pointcloud files. Some `.txt` files are created, and stored in `examples/output`.

Main `Python` scripts:
    
* `image_browser.py` : Find pairs of images based on the distance between both frames and their similiraties in terms of tilt & (theta-pan) angle. Matches are written to a .txt file.
    * **There are 2 paths to edit to use this file:** 
        * *dataset_dir*, which indicates where the dataset folder is located,
        * *image_auxilliary*, which indicates where the image_auxilliary.csv file is located. 
        * There is an entry point `if __name__ == '__main__':` at the bottom of the file, and the main function to use is `group_images_baseline()`, which creates pairs of images based on the distance between the 2 view points (0.97 <= dist <= 1.03), the tilt difference (< 0.01, to avoid sky pictures), and the (theta-pan) difference, which should be limited to ~20°.
        * The result is a `.txt` file containing the filepaths to the matched images. This file is used by `DeMoN.py` to create corresponding pointclouds. Experiments show that for a 40k images dataset, we match ~1100 pairs.
        * This script also computes the average x,y coordinates and store them to file (*will be used later for pointclouds visualization*). It also stores the computed distances to file (*will be used as a scale factor for visualization*).

* `DeMoN.py` : Class used to load the network once and pass it a list of images pairs in one forward pass. Creates the corresponding pointclouds in `pointclouds/` (also saves thresholded poinctlouds in `pointclouds_thres/`).
    * **There are 2 paths to edit to use this file:**
        * *pointclouds_dir*, which indicates where the created pointclouds are stored,
        * *pointclouds_thres_dir*, which indicates where the created thresholded pointclouds are stored.
    * **How to use**: First create & load the network by `demon = DeMoN()`, then use `demon.run_many('image_pairs.txt')` to send pairs of images to the network in one pass. `image_pairs.txt` is created by `image_browser.py`. 

* `pc_lake_position.py` : Create the visualization of a specified number of pointclouds on the lake plan.
    * **There are 3 paths to edit to use this file:**
        * *image_auxilliary*: indicates where image_auxilliary.csv is located
        * *center_coord*: filepath to where are stored average x,y coordinates.
        * *distances*: filepath where are stored distances between the 2 images of each pair.
    * The parameter `z_limit` indicates the depth threshold used to ignore points above that limit.
    * The script entry point is at the bottom of the file, and the main function is `visualization()`.
    * This file also defines some helpers method, e.g. to construct a rotation matrix, reading a list into tuples, fetching parameter values in `ìmage_auxilliary.csv` etc.

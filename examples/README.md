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

The code written is located in `demon/examples` and some functions are located in `demon/python/depthmotionnet/vis.py`

Main `Python` scripts:
* `image_browser.py` : Find pairs of images based on the distance between both frames and their similiraties in terms of tilt & (theta-pan) angle. Matches are written to a .txt file.
* `DeMoN.py` : Class used to load the network once and pass it a list of images pairs in one forward pass. Creates the corresponding pointclouds in `pointclouds/` (also saves thresholded poinctlouds in `pointclouds_thres/`).
* `pc_lake_position.py` : Create the visualization of a specified number of pointclouds on the lake plan.

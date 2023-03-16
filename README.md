# MultiMap3D

This repository contains code to build a Multi-level perceptual semantic map in real-time.

 We use [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2) as the slam part and [FCAF3D](https://github.com/SamsungLabs/fcaf3d) to achieve point cloud detection. we have to mention that our orb-slam2's work also based this work->[ORB-SLAM2_DENSE](https://github.com/IATBOMSW/ORB-SLAM2_DENSE).



## 1.BUILDING

move to your ros workspace and get the code as rospackage.

```shell
cd /${your_workspace}/src
git clone https://github.com/ZhouJankin/MultiMap3D.git
```

### orb-slam2 requirements

[Pangolin](https://github.com/stevenlovegrove/Pangolin)、 [OpenCV](http://opencv.org)、[Eigen3](http://eigen.tuxfamily.org)、DBoW2 and g2o(included in Thirdparty folder)

### build orb-slam2:

```shell
cd MultiMap3D/ORB-SLAM2_DENSE-master
./build.sh
cd Examples/ROS/ORB_SLAM2_DENSE
mkdir build
cd build
cmake ..
make -j4
```

### build mmdetection3d

create a conda environment

```shell
conda create -n Multimap3D python=3.8

```

Then install the pytorch and cuda.(we use pytorch 1.9 and cuda 11.1 with a nivdia RTX 3070)

Follow the instruction and install the [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

if you wanna use FCAF3D as detection method as well, then you need to install [MinkowskiEngine](https://nvidia.github.io/MinkowskiEngine/quick_start.html)

```shell
pip3 install -U MinkowskiEngine

```

if it dosen't work, check the website above and find another way.

then download the checkpoints you need in checkpoints file

```shell
cd ${your_workspace}/src/MultiMap3D/checkpoints
```

download here <https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fcaf3d>

## 2.Using

Before running your package , don't forget catkin_make if you had any changes, and source your workspace.

```shell
(optional) cd ${your_workspace}  && catkin_make
source devel/setup.bash
```

### First run the detection and database

The detection and database code are in [fcaf3d_sort_demo.py](https://github.com/ZhouJankin/MultiMap3D/blob/main/ORB-SLAM2_DENSE-master/Examples/ROS/ORB_SLAM2_DENSE/scripts/fcaf3d_sort_demo.py)

Change the checkpoints file and config files' path in fcaf3d_sort_demo.py

```pytho
config_file = '${your_workspace}/src/MultiMap3D/configs/fcaf3d/fcaf3d_8x2_scannet-3d-18class.py'
checkpoint_file = '${your_workspace}/src/MultiMap3D/checkpoints/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth'

```

And don't forget to change class names based your model.

Then choose use local point cloud(/keyframe_cloud) to detect or the global point cloud(/cloud2)

```pyth
rospy.Subscriber('/keyframe_cloud', PointCloud2, callback)
or rospy.Subscriber('/cloud2', PointCloud2, callback)
```

run fcaf3d_sort_demo.py

```shell
python ${your_workspace}/src/MultiMap3D/ORB-SLAM2_DENSE-master/Examples/ROS/ORB_SLAM2_DENSE/scripts/fcaf3d_sort_demo.py
```

It needs time to load the checkpoint file, when you see 'start', you can run the slam part.

### run the slam part

```shell
roslaunch orb_slam2_dense tum_pioneer.launch
```

you can change your dataset at ${your_workspace}/src/MultiMap3D/ORB-SLAM2_DENSE-master/Examples/ROS/ORB_SLAM2_DENSE/launch/tum_pioneer.launch

and change  other parameters in ${your_workspace}/src/MultiMap3D/ORB-SLAM2_DENSE-master/Examples/ROS/ORB_SLAM2_DENSE/params


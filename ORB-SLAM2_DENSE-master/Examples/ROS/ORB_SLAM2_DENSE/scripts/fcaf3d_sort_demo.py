#! /media/zhou/0EE2C649E2C634AD/anaconda3/envs/openmmlab/bin/python

# Copyright (c) OpenMMLab. All rights reserved.
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import ctypes
import struct
import math
from mySORT_3d import *

from argparse import ArgumentParser
from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
from mmdet3d.core.points import get_points_type
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray

from room_classifier.room_classifier import RoomClassifier

rc = RoomClassifier()

# 配置模型
# 初始化Sort维护tracker
mot_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)

# config_file = '/media/zhou/0EE2C649E2C634AD/mmdetection3d/configs/fcaf3d/fcaf3d_8x2_sunrgbd-3d-10class.py'
# checkpoint_file = '/media/zhou/0EE2C649E2C634AD/mmdetection3d/checkpoints/fcaf3d_8x2_sunrgbd-3d-10class_20220805_165017.pth'

config_file = '/media/zhou/0EE2C649E2C634AD/mmdetection3d/configs/fcaf3d/fcaf3d_8x2_scannet-3d-18class.py'
checkpoint_file = '/media/zhou/0EE2C649E2C634AD/mmdetection3d/checkpoints/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

# sunrgbd class
# class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
#                'night_stand', 'bookshelf', 'bathtub')

# scannet class
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

score_thr = 0.165

pub_box_3D = rospy.Publisher('/Boxes_3D', BoundingBoxArray, queue_size=10)
pub_label_3D = rospy.Publisher('/label_3D', MarkerArray, queue_size=10)

pub_box_grid = rospy.Publisher('/Boxes_grid', BoundingBoxArray, queue_size=10)
pub_label_grid = rospy.Publisher('/label_grid', MarkerArray, queue_size=10)

pub_box_object = rospy.Publisher('/Boxes_object', BoundingBoxArray, queue_size=10)
pub_label_object = rospy.Publisher('/label_object', MarkerArray, queue_size=10)

def callback(data):
    # 创建array
    xyzrgb = np.array([[0,0,0,0,0,0]])
    # 从message中提取数据
    gen = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z", "rgb"))
    int_data = list(gen)

    # 处理rgb信息
    for x in int_data:
        test = x[3]         # rgb信息
        # cast float32 to int so that bitwise operation are possible
        s = struct.pack('>f', test)
        i = struct.unpack('>l', s)[0]
        # you can get back the float value by inverse operations
        pack = ctypes.c_uint32(i).value
        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)
        # 于是rgb信息就变成 0-255 range了
        xyzrgb = np.append(xyzrgb, [[x[0],x[1],x[2],r,g,b]], axis=0)

    # point_cloud = np.array(list(pc2.read_points(data, skip_nans=True, field_names=("x","y","z","rgb"))))
    # print(xyzrgb[:10])

    # 将数据输入mmdet3d进行推理
    points_class = get_points_type('DEPTH')
    points_mmdet3d = points_class(xyzrgb, points_dim=xyzrgb.shape[-1], attribute_dims=None)
    result, data = inference_detector(model, points_mmdet3d)

    boxes = result[0]['boxes_3d'].tensor.cpu().numpy()
    label = result[0]['labels_3d'].numpy()
    score = result[0]['scores_3d'].numpy()



    # 创建一个Box array grid_level并把box放入
    boxes_grid = BoundingBoxArray()
    boxes_grid.header.stamp = rospy.Time.now()
    # 选择坐标系
    boxes_grid.header.frame_id = 'grid'
    # 创建marker_array
    markers_grid = MarkerArray()

    # 创建一个Box array map_level 并把box放入
    boxes_3D = BoundingBoxArray()
    boxes_3D.header.stamp = rospy.Time.now()
    # 选择坐标系
    boxes_3D.header.frame_id = 'map'
    # 创建marker_array
    markers_3D = MarkerArray()

    # 创建一个Box array object_level 并把box放入
    boxes_object = BoundingBoxArray()
    boxes_object.header.stamp = rospy.Time.now()
    # 选择坐标系
    boxes_object.header.frame_id = 'object'
    # 创建marker_array
    markers_object = MarkerArray()

    # 正式程序
    # 判断结果分数,大于阈值的交给mot_tracker进行数据库维护
    dets = []
    for i in range(len(score)):
        if score[i] > score_thr:
            dets.append(np.concatenate((boxes[i], [label[i]], [score[i]])).reshape(1, -1))
    dets = np.concatenate(dets)
    # track_bbs_ids - a numpy array of detections in the format [[x,y,z,l,w,h,yaw,label,score, id],[x,y,z,l,w,h,yaw,label,score, id],...]
    track_bbs_ids = mot_tracker.update(dets)
    
    # *********************************************************************************************For Arturs  here
    
    # get_my_labelArray = track_bbs_ids[..., 7].reshape(-1,1)  # or whatever shape
    
    objects = [obj[7] for obj in track_bbs_ids]
    
    obj_names = [class_names[i.astype(int)] for i in objects]
    
    # now we'll get the objects into a string separated by a space
    objs_in_room_as_string = ""
    for obj in obj_names:
        objs_in_room_as_string += obj + " "
        
    objs_in_room_as_string = objs_in_room_as_string[:-1]
    
    room_type = rc.predict(objs_in_room_as_string)

    # marker = Marker()
    # marker.header.frame_id = 'camera_optical'
    # marker.action = Marker.ADD
    # marker.id = -1
    # marker.type = Marker.TEXT_VIEW_FACING
    # marker.text = room_type
    # marker.pose.position.x = 0.0
    # marker.pose.position.y = 0.0
    # marker.pose.position.z = 10.0
    # marker.pose.orientation.w = 1.0
    # marker.scale.z = 0.2
    # marker.color.r = 1.0
    # marker.color.g = 1.0
    # marker.color.b = 1.0
    # marker.color.a = 1.0
    # marker_array.markers.append(marker)
    
    # *********************************************************************************************
    id = 0
    for object in track_bbs_ids:
            # 创建jsk_recognition_msg中的BoundingBox
            box_3D = BoundingBox()
            box_3D.header.stamp = rospy.Time.now()
            box_3D.header.frame_id = 'map'
            box_3D.pose.position.x = object[0]
            box_3D.pose.position.y = object[1]
            box_3D.pose.position.z = object[2] + object[5]/2

            cy = math.cos(object[6] * 0.5)
            sy = math.sin(object[6] * 0.5)
            cp = math.cos(0 * 0.5)
            sp = math.sin(0 * 0.5)
            cr = math.cos(0 * 0.5)
            sr = math.sin(0 * 0.5)

            box_3D.pose.orientation.x = cy * cp * sr - sy * sp * cr
            box_3D.pose.orientation.y = sy * cp * sr + cy * sp * cr
            box_3D.pose.orientation.z = sy * cp * cr - cy * sp * sr
            box_3D.pose.orientation.w = cy * cp * cr + sy * sp * sr
            box_3D.dimensions.x = object[3]
            box_3D.dimensions.y = object[4]
            box_3D.dimensions.z = object[5]
            box_3D.label = object[7].astype(int)
            box_3D.value = 0.8

            box_grid = BoundingBox()
            box_grid.header.stamp = rospy.Time.now()
            box_grid.header.frame_id = 'grid'
            box_grid.pose = box_3D.pose
            box_grid.dimensions = box_3D.dimensions
            box_grid.label = box_3D.label
            box_grid.value = 0.5

            box_object = BoundingBox()
            box_object.header.stamp = rospy.Time.now()
            box_object.header.frame_id = 'object'
            box_object.pose = box_3D.pose
            box_object.dimensions = box_3D.dimensions
            box_object.label = box_3D.label
            box_object.value = 0.9


            # # new markerbox
            # marker = Marker()
            # marker.header.frame_id = 'map'
            # marker.action = Marker.ADD
            # marker.id = id
            # marker.lifetime = rospy.Duration()
            # marker.type = Marker.CUBE
            #
            # marker.pose = box.pose
            # marker.scale = box.dimensions
            # marker.color.r = 1.0
            # marker.color.g = 1.0
            # marker.color.b = 1.0
            # marker.color.a = 0.4



            # marker
            marker_3D = Marker()
            marker_3D.header.frame_id = box_3D.header.frame_id
            marker_3D.action = Marker.ADD
            marker_3D.id = id
            # 需要设置自动消除时间，否则marker会一直留在rviz
            # marker.lifetime = rospy.Duration(1.0)
            marker_3D.type = Marker.TEXT_VIEW_FACING
            #打上标签label
            marker_3D.text = class_names[object[7].astype(int)] + str(object[9].astype(int))
            marker_3D.pose.position = box_3D.pose.position
            marker_3D.scale.z = 0.2
            marker_3D.color.r = 1.0
            marker_3D.color.g = 1.0
            marker_3D.color.b = 1.0
            marker_3D.color.a = 1.0

            #marker_grid
            marker_grid = Marker()
            marker_grid.header.frame_id = box_grid.header.frame_id
            marker_grid.action = Marker.ADD
            marker_grid.id = id
            marker_grid.type = Marker.TEXT_VIEW_FACING
            marker_grid.text = marker_3D.text
            marker_grid.pose = marker_3D.pose
            marker_grid.scale.z = 0.2
            marker_grid.color = marker_3D.color

            # marker_object
            marker_object = Marker()
            marker_object.header.frame_id = box_object.header.frame_id
            marker_object.action = Marker.ADD
            marker_object.id = id
            marker_object.type = Marker.TEXT_VIEW_FACING
            marker_object.text = marker_3D.text
            marker_object.pose = marker_3D.pose
            marker_object.scale.z = 0.2
            marker_object.color = marker_3D.color


            # put boxes and labels in map_level
            boxes_3D.boxes.append(box_3D)
            markers_3D.markers.append(marker_3D)
            # put boxes and labels in grid_level
            boxes_grid.boxes.append(box_grid)
            markers_grid.markers.append(marker_grid)
            # put boxes and labels in object_level
            boxes_object.boxes.append(box_object)
            markers_object.markers.append(marker_object)


            id = id + 1
            # box_array.labels.append('example')

    # 发布该话题
    pub_box_3D.publish(boxes_3D)
    print(len(markers_3D.markers))
    pub_label_3D.publish(markers_3D)

    pub_box_grid.publish(boxes_grid)
    pub_label_grid.publish(markers_grid)

    pub_box_object.publish(boxes_object)
    pub_label_object.publish(markers_object)


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/keyframe_cloud', PointCloud2, callback)
    # rospy.Subscriber('/cloud2', PointCloud2, callback)
    # pub = rospy.Publisher('/result', BoundingBoxArray, queue_size=10)
    # 接收点云的频率
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ =='__main__':
    print('start')
    listener()


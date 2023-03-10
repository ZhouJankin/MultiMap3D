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



mot_tracker = Sort(max_age=3, min_hits=2, iou_threshold=0.3)

# 配置模型
config_file = '/media/zhou/0EE2C649E2C634AD/mmdetection3d/configs/fcaf3d/fcaf3d_8x2_scannet-3d-18class.py'
checkpoint_file = '/media/zhou/0EE2C649E2C634AD/mmdetection3d/checkpoints/fcaf3d_8x2_scannet-3d-18class_20220805_084956.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')
# class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
#                'night_stand', 'bookshelf', 'bathtub')
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')

score_thr = 0.2

pub = rospy.Publisher('/Boxes_result', BoundingBoxArray, queue_size=10)
label_pub = rospy.Publisher('/label_result', MarkerArray, queue_size=10)

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



    # 创建一个Box array并把box放入
    box_array = BoundingBoxArray()
    box_array.header.stamp = rospy.Time.now()
    # 选择坐标系
    box_array.header.frame_id = 'map'
    # 创建marker_array
    marker_array = MarkerArray()


    # # box测试部分 注释
    # box = BoundingBox()
    # box.header.stamp = rospy.Time.now()
    # box.header.frame_id = 'map'
    # box.pose.position.x = 1.0
    # box.pose.position.y = 1.5
    # box.pose.position.z = 2.0
    #
    # cy = math.cos(0.7 * 0.5)
    # sy = math.sin(0.7 * 0.5)
    # cp = math.cos(0 * 0.5)
    # sp = math.sin(0 * 0.5)
    # cr = math.cos(0 * 0.5)
    # sr = math.sin(0 * 0.5)
    #
    # box.pose.orientation.x = cy * cp * sr - sy * sp * cr
    # box.pose.orientation.y = sy * cp * sr + cy * sp * cr
    # box.pose.orientation.z = sy * cp * cr - cy * sp * sr
    # box.pose.orientation.w = cy * cp * cr + sy * sp * sr
    # box.dimensions.x = 1.0
    # box.dimensions.y = 1.5
    # box.dimensions.z = 2.0
    # box.label = 1
    # box.value = 0.5
    #
    # # marker测试 通过
    # marker = Marker()
    # marker.header.frame_id = box.header.frame_id
    # marker.type = Marker.TEXT_VIEW_FACING
    # marker.text = 'label_1'
    # marker.pose.position = box.pose.position
    # marker.scale.z = 0.2
    # marker.color.r = 1.0
    # marker.color.g = 1.0
    # marker.color.b = 1.0
    # marker.color.a = 1.0
    #
    # box_array.boxes.append(box)
    # marker_array.markers.append(marker)


    # 正式程序
    # 判断结果分数,大于阈值的发送话题显示
    # dets = []
    # for i in range(len(score)):
    #     if score[i] > score_thr:
    #         dets.append(np.concatenate((boxes[i], [label[i]], [score[i]])).reshape(1, -1))
    # dets = np.concatenate(dets)
    # track_bbs_ids = mot_tracker.update(dets)

    # for object in track_bbs_ids:
    for i in range(len(score)):
        if score[i] > score_thr:
            # 创建jsk_recognition_msg中的BoundingBox
            box = BoundingBox()
            box.header.stamp = rospy.Time.now()
            box.header.frame_id = 'map'
            box.pose.position.x = boxes[i][0]
            box.pose.position.y = boxes[i][1]
            box.pose.position.z = boxes[i][2] + boxes[i][5]/2

            cy = math.cos(boxes[i][6] * 0.5)
            sy = math.sin(boxes[i][6] * 0.5)
            cp = math.cos(0 * 0.5)
            sp = math.sin(0 * 0.5)
            cr = math.cos(0 * 0.5)
            sr = math.sin(0 * 0.5)

            box.pose.orientation.x = cy * cp * sr - sy * sp * cr
            box.pose.orientation.y = sy * cp * sr + cy * sp * cr
            box.pose.orientation.z = sy * cp * cr - cy * sp * sr
            box.pose.orientation.w = cy * cp * cr + sy * sp * sr
            box.dimensions.x = boxes[i][3]
            box.dimensions.y = boxes[i][4]
            box.dimensions.z = boxes[i][5]
            box.label = label[i]
            box.value = 0.5

            # marker
            marker = Marker()
            marker.header.frame_id = box.header.frame_id
            marker.action = Marker.ADD
            marker.id = i
            # 需要设置自动消除时间，否则marker会一直留在rviz
            marker.lifetime = rospy.Duration(5.0)
            marker.type = Marker.TEXT_VIEW_FACING
            marker.text = class_names[label[i]]
            marker.pose.position = box.pose.position
            marker.scale.z = 0.2
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            box_array.boxes.append(box)
            marker_array.markers.append(marker)
            # box_array.labels.append('example')

    # 发布该话题
    pub.publish(box_array)
    print(len(marker_array.markers))
    label_pub.publish(marker_array)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/keyframe_cloud', PointCloud2, callback)
    # pub = rospy.Publisher('/result', BoundingBoxArray, queue_size=10)
    # 接收点云的频率
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        rate.sleep()


if __name__ =='__main__':
    print('start')
    listener()

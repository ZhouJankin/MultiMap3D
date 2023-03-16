_base_ = ['fcaf3d_8x2_scannet-3d-18class.py']
n_points = 100000   # 最大点云数量(?)

model = dict(
    head=dict(
        n_classes=10, n_reg_outs=8, bbox_loss=dict(type='RotatedIoU3DLoss')))

dataset_type = 'SUNRGBDDataset'     # 数据集类型
data_root = 'data/sunrgbd/'         # 数据路径
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')

# 后面定义了新的train_pipeline/test_pipeline 然后将其传入data
train_pipeline = [      # 训练流水线
    dict(
        type='LoadPointsFromFile',      # 第一个流程，用于读取点，更多细节请参考 mmdet3d.datasets.pipelines.indoor_loading
        coord_type='DEPTH',         # 是否使用变换高度
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),       # 使用所读取点的哪些维度
    dict(type='LoadAnnotations3D'),        # 第二个流程，用于读取标注，更多细节请参考 mmdet3d.datasets.pipelines.indoor_loading
    dict(type='PointSample', num_points=n_points),      # 室内点采样，更多细节请参考 mmdet3d.datasets.pipelines.indoor_sample
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5),    # 随机地翻折图像
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
                flip_ratio_bev_vertical=0.5),
            dict(type='PointSample', num_points=n_points),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=False, use_lidar=True),
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

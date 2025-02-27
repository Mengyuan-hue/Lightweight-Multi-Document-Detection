_base_ = ['mmpose::_base_/default_runtime.py']

# 数据集类型及路径
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = r'E:\XuyuanFiles\csqHelpProject\keypoint_rcnn_training_pytorch\Mpaper_data_my\Mpaper_data_my'

# 纸张关键点检测数据集-元数据
dataset_info = dict(

    dataset_name='coco',
    paper_info=dict(
        author='Vincent Hu',
        title='MpaperData for coco',
        container='no',
        year='2023',
        homepage='no',
    ),
    keypoint_info={
        0:
        dict(name='LT', id=0, color=[0, 0, 255], type='corn', swap=''),
        1:
        dict(
            name='RT',
            id=1,
            color=[0, 255, 0],
            type='corn',
            swap=''),
        2:
        dict(
            name='RD',
            id=2,
            color=[255, 0, 0],
            type='corn',
            swap=''),
        3:
        dict(
            name='LD',
            id=3,
            color=[255, 255, 255],
            type='corn',
            swap='')
    },
    skeleton_info={
        0:
        dict(link=('LT', 'RT'), id=0, color=[255, 255, 0]),
        1:
        dict(link=('RT', 'RD'), id=1, color=[255, 255, 0]),
        2:
        dict(link=('RD', 'LD'), id=2, color=[255, 255, 0]),
        3:
        dict(link=('LD', 'LT'), id=3, color=[255, 255, 0]),
    },
    joint_weights=[
        1., 1., 1., 1.
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025
    ])

# 获取关键点个数


# 训练超参数
max_epochs = 100 # 训练 epoch 总数
val_interval = 10 # 每隔多少个 epoch 保存一次权重文件
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 16
val_batch_size = 8
stage2_num_epochs = 0
base_lr = 4e-3
randomness = dict(seed=21)

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 学习率
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]
# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
# codec = dict(
#     type='SimCCLabel',
#     input_size=(256, 256),
#     sigma=(12, 12),
#     simcc_split_ratio=2.0,
#     normalize=False,
#     use_dark=False)
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(32, 32), sigma=2)
# 不同输入图像尺寸的参数搭配
# input_size=(256, 256),
# sigma=(12, 12)
# in_featuremap_size=(8, 8)
# input_size可以换成 256、384、512、1024，三个参数等比例缩放
# sigma 表示关键点一维高斯分布的标准差，越大越容易学习，但精度上限会降低，越小越严格，对于人体、人脸等高精度场景，可以调小，RTMPose 原始论文中为 5.66

# 不同模型的 config： https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose/rtmpose/body_2d_keypoint

# 模型：CPM
# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CPM',
        in_channels=3,
        out_channels=4,
        feat_channels=64,
        num_stages=6),
    head=dict(
        type='CPMHead',
        in_channels=4,
        out_channels=4,
        num_stages=6,
        deconv_out_channels=None,
        final_layer=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]


# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='Train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='Test.json',
        data_prefix=dict(img=''),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

default_hooks = {
    'checkpoint': {'save_best': 'PCK','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

# evaluators
val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + '\\Test.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1])
]

test_evaluator = val_evaluator



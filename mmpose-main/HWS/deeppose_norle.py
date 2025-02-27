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
NUM_KEYPOINTS = len(dataset_info['keypoint_info'])
dataset_info['joint_weights'] = [1.0] * NUM_KEYPOINTS
dataset_info['sigmas'] = [0.025] * NUM_KEYPOINTS

# 训练超参数
max_epochs = 200 # 训练 epoch 总数
val_interval = 10 # 每隔多少个 epoch 保存一次权重文件
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 32
val_batch_size = 16
stage2_num_epochs = 0
base_lr = 4e-3
randomness = dict(seed=21)

# 优化器
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,
))
# 学习率
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=train_cfg['max_epochs'],
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]


# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
# codec = dict(
#     type='SimCCLabel',
#     input_size=(256, 256),
#     sigma=(12, 12),
#     simcc_split_ratio=2.0,
#     normalize=False,
#     use_dark=False)
codec = dict(type='RegressionLabel', input_size=(256, 256))

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.,
        out_indices=(7, ),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/top_down/'
            'mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='RegressionHead',
        in_channels=1280,
        num_joints=4,
        loss=dict(type='SmoothL1Loss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        shift_coords=True,
    ),
)


backend_args = dict(backend='local')
# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dataset_info,
        ann_file='Train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dataset_info,
        ann_file='Test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

default_hooks = {
    'checkpoint': {'save_best': 'PCK','rule': 'greater'},
    'logger': {'interval': 1}
}

# custom_hooks = [
#     dict(
#         type='EMAHook',
#         ema_type='ExpMomentumEMA',
#         momentum=0.0002,
#         update_buffers=True,
#         priority=49),
#     dict(
#         type='mmdet.PipelineSwitchHook',
#         switch_epoch=max_epochs - stage2_num_epochs,
#         switch_pipeline=train_pipeline)
# ]

# evaluators
val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + '\\Test.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1])
]

test_evaluator = val_evaluator



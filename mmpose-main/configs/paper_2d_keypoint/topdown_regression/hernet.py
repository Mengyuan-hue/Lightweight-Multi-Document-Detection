_base_ = ['mmpose::_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))

# learning policy
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
auto_scale_lr = dict(base_batch_size=512)



# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

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
        dict(link=('LD', 'RT'), id=3, color=[255, 255, 0]),
    },
    joint_weights=[
        1., 1., 1., 1.
    ],
    sigmas=[
        0.025, 0.025, 0.025, 0.025
    ])

# 获取关键点个数
NUM_KEYPOINTS = len(dataset_info['keypoint_info'])


# pipelines
train_pipeline = [
    dict(type='LoadImage'),
]
val_pipeline = [
    dict(type='LoadImage'),
]

# data loaders
train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='Train.json',
        data_prefix=dict(img='Train\\images'),
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
        ann_file='Test.json',
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='Test\\images'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

default_hooks = {
    'checkpoint': {'save_best': 'PCK','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]
# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
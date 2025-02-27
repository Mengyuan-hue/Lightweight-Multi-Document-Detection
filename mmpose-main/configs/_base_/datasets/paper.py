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
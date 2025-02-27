from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = r'E:/XuyuanFiles/mmlab/mmdeploy-main/configs/mmpose/pose-detection_onnxruntime_static.py'
model_cfg = r'E:\XuyuanFiles\mmlab\mmpose-main\work_dirs\deeppose_seatt_regression_cocopaper_256x256\deeppose_seatt_regression_cocopaper_256x256.py'
device = 'cpu'
backend_model = ['E:\XuyuanFiles\mmlab\onnx\end2end.onnx']
image = r'E:\XuyuanFiles\PaperDataSet\Train\images\1182.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output_pose1.png')
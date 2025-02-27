from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

img = r'E:\XuyuanFiles\PaperDataSet\Train\images\1133_horzontalflip.jpg'
work_dir = r'E:\XuyuanFiles\mmlab\onnx'
save_file = 'end2end.onnx'
deploy_cfg = 'mmdeploy-main/configs/mmpretrain/classification_onnxruntime_dynamic.py'
model_checkpoint=r'E:\XuyuanFiles\mmlab\mmdetection-main\work_dirs\yolox_D\best_coco_bbox_mAP_epoch_400.pth'
model_cfg=r'E:\XuyuanFiles\mmlab\mmdetection-main\work_dirs\yolox_D\yolox_D.py'
device = 'cpu'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
  model_checkpoint, device)

# 2. extract pipeline info for sdk use (dump-info)
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint, device=device)

E:\XuyuanFiles\mmlab\mmdeploy-main\configs\mmdet\detection\detection_onnxruntime_static.py
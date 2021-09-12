'''
@Author：Runsen
'''
import os
'''
yolov5m_coco.pt
'''
import torch

model = torch.load('data/models/yolov5m_coco.pt', map_location=torch.device('cuda'))
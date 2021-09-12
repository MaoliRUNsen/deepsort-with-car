import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size
from utils.datasets import letterbox
from utils.general import  scale_coords

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config


def img_transfer(frame):
    # Padded resize
    img = letterbox(frame, p.imgsz, stride=p.stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # tran
    img = torch.from_numpy(img).to(p.device)
    img = img.half() if p.half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def get_results(pred, img, frame_shape, names):
    ''' ***** param explain *****
        :param pred: infer + nms results
        :param img: torch infer array, shape= [x, 3, w, h], x(axis=0) is array number
        :param frame: origin image shape= [w_org, h_org, 3]
        :param names: class names
        :return: standard api output
    '''

    rst = []
    for i, det in enumerate(pred):  # detections per image

        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame_shape).round()
        for *xyxy, conf, cls in reversed(det):
            obj = {}

            score = float(f"{conf:.2f}")
            if score < 0.5: continue

            obj["class_name"] = names[int(cls)]
            obj["score"] = score
            obj["bbox"] = [(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))]
            rst.append(obj)

    return rst


def get_boxes(pred, img, frame_shape, names):
    ''' ***** param explain *****
        :param pred: infer + nms results
        :param img: torch infer array, shape= [x, 3, w, h], x(axis=0) is array number
        :param frame_shape: origin image shape= [w_org, h_org, 3]
        :param names: class names
        :return: standard bbox information output
    '''
    boxes = []
    for det in pred:
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame_shape).round()

        for *x, conf, cls_id in det:
            lbl = names[int(cls_id)]
            if lbl not in ['bicycle', 'car', 'motorcycle', 'bus', 'truck']:
                continue
            x1, y1 = int(x[0]), int(x[1])
            x2, y2 = int(x[2]), int(x[3])
            boxes.append((x1, y1, x2, y2, lbl, conf))
    return boxes


def draw_bbox(image, bboxes):
    image_h, image_w = image.shape[:2]
    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for obj in bboxes:
        class_name = obj['class_name']
        score = obj['score']
        bbox = obj['bbox']

        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[1][0]), int(bbox[1][1])

        bbox_mess = '%s: %.2f' % (class_name, score)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), bbox_thick)  # filled
        cv2.putText(image, bbox_mess, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (255, 0, 255), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image


def load_model(weights, device):
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model

    # 如果设备为gpu，使用Float16
    if half:
        model.half()  # to FP16
    return model, device, half


def load_deepsort(dp_file="./deep_sort/configs/deep_sort.yaml" ):
    cfg = get_config(config_file=None)
    cfg.merge_from_file(dp_file)

    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    return deepsort


def search_label(center_x, center_y, bboxes_xyxy, max_dist_threshold):
    """
    在 yolov5 的 bbox 中搜索中心点最接近的label
    :param center_x:
    :param center_y:
    :param bboxes_xyxy:
    :param max_dist_threshold:
    :return: 字符串
    """
    label = ''
    # min_label = ''
    min_dist = -1.0

    for x1, y1, x2, y2, lbl, conf in bboxes_xyxy:
        center_x2 = (x1 + x2) * 0.5
        center_y2 = (y1 + y2) * 0.5

        # 横纵距离都小于 max_dist
        min_x = abs(center_x2 - center_x)
        min_y = abs(center_y2 - center_y)

        if min_x < max_dist_threshold and min_y < max_dist_threshold:
            # 距离阈值，判断是否在允许误差范围内
            # 取 x, y 方向上的距离平均值
            avg_dist = (min_x + min_y) * 0.5
            if min_dist == -1.0:
                # 第一次赋值
                min_dist = avg_dist
                # 赋值label
                label = lbl
                pass
            else:
                # 若不是第一次，则距离小的优先
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    # label
                    label = lbl
                pass
            pass
        pass

    return label


def draw_bboxes(image, bboxes, line_thickness):
    '''
    实现画方框的功能
    :param image: 图片
    :param bboxes: 四个位置
    :param line_thickness:线的厚度
    :return:
    '''
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    list_pts = []
    point_radius = 4

    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        color = (0, 255, 0)

        # 撞线的点
        check_point_x = int((x1 + x2) /2 )
        check_point_y = y2

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = np.array(list_pts, np.int32)

        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))

        list_pts.clear()

    return image


def track_update(bboxes, image, deepsort):
    bbox_xywh = []
    confs = []
    bboxes2draw = []

    if len(bboxes) > 0:
        for x1, y1, x2, y2, lbl, conf in bboxes:
            obj = [
                int((x1 + x2) * 0.5), int((y1 + y2) * 0.5),
                x2 - x1, y2 - y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, image)

        for x1, y1, x2, y2, track_id in list(outputs):
            # x1, y1, x2, y2, track_id = value
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            label = search_label(center_x=center_x, center_y=center_y,
                                 bboxes_xyxy=bboxes, max_dist_threshold=20.0)

            bboxes2draw.append((x1, y1, x2, y2, label, track_id))

    return bboxes2draw


class Params(object):
    def __init__(self):

        dic_infer = {
            "img_size": 640,
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "device": "0"
        }

        # dic_infer["device"] = "cpu"
        # weights = "data/models/yolov5l_210714.pt"
        weights = "data/models/yolov5m_coco.pt"

        # laod  models
        model, device, half = load_model(weights, dic_infer["device"])
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(dic_infer["img_size"], s=stride)  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names

        # load deep sort
        dp_file = "./deep_sort/configs/deep_sort.yaml"
        deepsort = load_deepsort(dp_file)

        self.model      = model
        self.half       = half
        self.device     = device
        self.imgsz      = imgsz
        self.stride     = stride
        self.img_size   = dic_infer["img_size"]
        self.conf_thres = dic_infer["conf_thres"]
        self.iou_thres  = dic_infer["iou_thres"]
        self.augment    = False  # 增大推理时间，默认为否
        self.classes    = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False   # agnostic_nms
        self.names = names
        self.deepsort = deepsort
        self.quene_name = 'flow.judge'

        # self.ip = '127.0.0.1'
        # self.src = r'data/videos/test14.mp4'
        # self.ip = '172.19.65.50'
        # self.src = r'rtsp://admin:Admin12345@172.31.20.191:554/Streaming/Channels/101?transportmode=unicast'


p = Params()

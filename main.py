from config import *
from util import *
import time
from mq import Producer
import json

from utils.general import non_max_suppression


def flow_detect(url):
    dic_sel = {}
    dic_all = {}
    record_list = []

    cap = cv2.VideoCapture(url)
    fps = cap.get(5)
    print('检测信息打印-----------\n\tip:{}\n\tfps:{}'.
          format(url, fps))

    max_all = 100
    max_sel = 40
    max_record = 40
    ip = p.ip
    mq_p = Producer(ip, 'guest', 'guest')

    while cap.isOpened():
        _, frame = cap.read()
        if frame is None: break

        # todo -->  1. inference  get standard  "boxes" format output
        img = img_transfer(frame)
        pred = p.model(img, augment=p.augment)[0]
        pred = non_max_suppression(pred, p.conf_thres, p.iou_thres, classes=p.classes, agnostic=p.agnostic_nms)
        boxes = get_boxes(pred, img, frame.shape, p.names)

        # todo -->  2. track object in choice area

        # todo -> 2.1 update dic_sel
        dic_sel = update_dic_sel(dic_all, dic_sel, cfg.value.n_all, cfg.point.choice_area)

        # todo -> 2.2 update dic_all + dic_sel
        if len(boxes) > 0:
            list_bboxs = track_update(boxes, frame, p.deepsort)     # spend time ~= 0.003796 second
            # output_im = draw_bboxes(frame, list_bboxs, line_thickness=None)     # spend time ~= 0.00005 second

            if len(list_bboxs) > 0:
                for bboxs in list_bboxs:
                    track_id = bboxs[5]
                    dic_all[track_id] = value_append(dic_all[track_id], bboxs[:4], cfg.value.n_all) \
                        if track_id in dic_all.keys() else [bboxs[:4]]

                    if track_id in dic_sel.keys():
                        dic_sel[track_id] = value_append(dic_sel[track_id], bboxs[:4], cfg.value.n_sel)

        # todo -> 2.3 analysis_rule and send message
        code, record_list = analysis_rule(dic_sel, record_list, fps)
        if code == "10001" or code == "10002":
            msg = {
                "timestamp": time.time(),
                "condition": code
            }
            mq_p.send_msg(p.quene_name,  json.dumps(msg))
            print('send:', msg)

        # todo -> 2.4 update all data, keep health memory condition
        if len(dic_all) >= max_all and len(bboxs)> 0:
            dic_all = update_save_dic(dic_all, list_bboxs, max_all)
        if len(dic_sel) >= max_sel and len(bboxs)> 0:
            dic_sel = update_save_dic(dic_sel, list_bboxs, max_sel)
        if len(record_list) >= max_record and len(bboxs)> 0:
            record_list = update_save_list(record_list, list_bboxs, max_record)


if __name__ == '__main__':
    flow_detect(p.src)

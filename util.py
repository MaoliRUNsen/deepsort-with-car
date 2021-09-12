from easydict import EasyDict
import math


'''
策略：
1）实时获得画面中每个ID的坐标列表dic_all(记录2秒信息)，并实时更新：剔除不在list_bboxs中的track_id

2）对候选区域dic_sel进行更新
 首先：剔除不在list_bboxs中的track_id
 接着：对 dic_bbox 中质点全部位于候选区域中
        a. 该ID不在dic_sel(记录15S信息)， 加入该字典
        b. 若该ID在dic_sel, 不做任何处理
      对 dic_bbox 中质点部分为于候选区域中，不做任何处理
      对 dic_bbox 中质点全部不位于候选区域中
        a. 该ID不在dic_sel， 不做任何处理
        b. 若该ID在dic_sel, 从dic_sel 剔除该ID

3）对 dic_sel部分进行如下逻辑判断：
    触碰线时，且在统计区域内待的时候小于多少秒认为是违规的
    对违规的，发送mq消息
        host: ***
        port: 5672
        topic: topic.mode.flow_judge
        message:{
            "image":
            "timestamp":
            "condition":"10001" # 10001 违规  10002 正常
        }

'''


def center_lines(p1, p2):
    assert p1[1] != p2[1]  # 不能平行y轴
    assert p1[0] != p2[0]  # 不能平行x轴

    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - k * p1[0]
    return k, b


cfg = EasyDict()
cfg.point = EasyDict()
cfg.point.choice_area = [(160, 430), (480, 760), (750, 750), (280, 430), (160, 430)]  # 逆时针
cfg.point.l1 = [(475, 740), (717, 729)]
cfg.point.l2 = [(394, 654), (595, 648)]
cfg.point.choice_area1 = [cfg.point.l1[0], cfg.point.l1[1], cfg.point.l2[1], cfg.point.l2[0], cfg.point.l1[0]]  # 逆时针
cfg.point.choice_area1_0 = [cfg.point.l1[0], cfg.point.l1[1], (626, 670), (426, 685), cfg.point.l1[0]]  # 逆时针, 0.5 area

cfg.line = EasyDict()
cfg.line.l1_k, cfg.line.l1_b = center_lines(cfg.point.l1[0], cfg.point.l1[1])

cfg.value = EasyDict()
cfg.value.n_all = 10
cfg.value.n_sel = 150
cfg.value.min_number = 2
cfg.value.min_second = 2.4
cfg.value.min_distance = 3  # 距离停止点最小px值


def distance_p2line(p, k, b):
    x0, y0 = p
    A, B, C = k, -1, b   # 一般直线方程 AX+BY+C = 0
    distance = abs(A*x0 + B*y0 + C) / math.sqrt(pow(A, 2) + pow(B, 2))
    return distance


def inner_polygon(point, rangelist):  # [[0,0],[1,1],[0,1],[0,0]] [1,0.8]
    # 判断是否在外包矩形内，如果不在，直接返回false
    lnglist = []
    latlist = []
    for i in range(len(rangelist) - 1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)
    if (point[0] > maxlng or point[0] < minlng or
            point[1] > maxlat or point[1] < minlat):
        return False

    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            # print("在顶点上")
            return False
        # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
            # 点在多边形边上
            if (point12lng == point[0]):
                return False
            if (point12lng < point[0]):
                count += 1
        point1 = point2
    if count % 2 == 0:
        return False
    else:
        return True


def value_append(lst_value, pst, max_len):
    value = lst_value.copy()
    if len(value) == max_len:
        value = value[-(max_len-1):]
        value.append(pst)
    else:
        value.append(pst)

    return value


def update_dic_sel(dic_all, dic_sel, n_all, point_list):
    for key in dic_all:     # 遍历每个ID
        values_a = dic_all[key]

        # todo -> 如果候选字典中存在该ID
        if key in dic_sel.keys():
            count_num = 0
            for a in values_a:
                p = ((a[0] + a[2]) // 2, a[3])

                # 如果所有质点不在候选区域内， 则从候选字典中删除该ID
                # 判断质点是否在候选区域内, 如果"dic_all"某个ID有任何一个在候选区域内 跳出内层循环
                if inner_polygon(p, point_list): break
                count_num += 1

            if count_num == n_all: dic_sel.pop(key)

        else:   # todo -> 如果候选字典中不存在该ID
            # 如果"dic_all"某个ID的坐标长度不够，不更新该字典
            if len(values_a) < n_all: continue

            count_num = 0
            for a in values_a:
                p = ((a[0] + a[2]) // 2, a[3])
                # 判断质点是否在候选区域内, 如果"dic_all"某个ID有任何一个不满足 跳出内层循环
                if not inner_polygon(p, point_list): break
                count_num += 1

            # 若质点个数全部在候选区域内，将在候选字典中增加该ID, 并赋值
            if count_num == n_all: dic_sel[key] = values_a

    return dic_sel


def analysis_rule(dic_sel, record_list, fps=25):
    code = "10000"

    for key in dic_sel:
        values = dic_sel[key]

        # 如果ID 在记录列表中，则
        if key in record_list:continue

        if len(values) < cfg.value.min_number:return code, record_list

        count_dist = 0      # 统计最新的坐标次数， 测试碰撞条件
        count_inner = 0
        count_inner_0 = 0
        for a in values[-cfg.value.min_number:]:
            p = ((a[0] + a[2]) // 2, a[3])
            distance = distance_p2line(p, cfg.line.l1_k, cfg.line.l1_b)

            if a[3] <= cfg.point.l1[0][1] and distance <= cfg.value.min_distance:
                count_dist += 1

            if count_dist == cfg.value.min_number:   # 当满足碰撞条件时候，统计在统计区域内的个数
                record_list.append(key)

                for s in values:
                    p1 = ((s[0] + s[2]) // 2, s[3])
                    if inner_polygon(p1, cfg.point.choice_area1):
                        count_inner += 1
                    if inner_polygon(p1, cfg.point.choice_area1_0):
                        count_inner_0 += 1

                # 当统计区域内个数小于某值，认为违规了
                if count_inner <= cfg.value.min_second * fps:
                    ratio = count_inner_0 / count_inner
                    if count_inner <= (cfg.value.min_second * 0.65 * fps):
                        code = "10001"
                    else:
                        # 如果测试部分比例占比超过 65 %， 我们认为它减速了
                        code = "10002" if ratio >= 0.65 else "10001"
                else:
                    code = "10002"

    return code, list(set(record_list))


def update_save_dic(dic,  list_bboxs, max_value):
    dic_new = {}
    max_val = max(max_value, len(dic))

    for idx, key in enumerate(dic):
        # 我们总是至少保留最后1/4的数据，其余的数据如果当前数据存在该ID我们不会删除
        if idx > max_val * 0.75:
            dic_new[key] = dic[key]

    for bboxs in list_bboxs:
        track_id = bboxs[5]
        if track_id in dic.keys():
            dic_new[track_id] = dic[track_id]

    return dic_new


def update_save_list(lst,  list_bboxs, max_val):
    new_lst = []
    max_val = max(max_val, len(lst))

    for idx, ID in enumerate(lst):
        # 我们总是至少保留最后1/4的数据，其余的数据如果当前数据存在该ID我们不会删除
        if idx > max_val * 0.75:
            new_lst.append(ID)

    for bboxs in list_bboxs:
        track_id = bboxs[5]
        if track_id in lst:
            new_lst.append(track_id)

    return list(set(new_lst))

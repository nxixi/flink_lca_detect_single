import numpy as np
import math


class LcaDetect(object):

    @staticmethod
    def from_map(conf_dict):
        id_field = conf_dict.get("id_field")
        time_field = conf_dict.get("time_field")
        value_field = conf_dict.get("value_field")
        abnormal_type_field = conf_dict.get("abnormal_type_field")
        value_his_field = conf_dict.get("value_his_field")
        train_res_field = conf_dict.get("train_res_field")
        detection_config_field = conf_dict.get("detection_config_field")
        day_start = conf_dict.get("day_start", 2)
        week_start = conf_dict.get("week_start", 14)
        agg = conf_dict.get("agg", 5)
        window = conf_dict.get("window", 1)
        sparse_period_thresh = conf_dict.get("sparse_period_thresh", 60)
        continuous_not_period_thresh = conf_dict.get("continuous_not_period_thresh", 7)
        his_window = conf_dict.get("his_window", 15)

        lca_detect = LcaDetect(id_field, time_field, value_field, abnormal_type_field, value_his_field, train_res_field,
                               detection_config_field, day_start=day_start, week_start=week_start, agg=agg,
                               window=window, sparse_period_thresh=sparse_period_thresh,
                               continuous_not_period_thresh=continuous_not_period_thresh, his_window=his_window)

        return lca_detect

    def __init__(self, id_field, time_field, value_field, abnormal_type_field, value_his_field, train_res_field,
                 detection_config_field, day_start=2, week_start=14, agg=5, window=1, sparse_period_thresh=60,
                 continuous_not_period_thresh=7, his_window=15):
        self.id_field = id_field
        self.time_field = time_field
        self.value_field = value_field
        self.abnormal_type_field = abnormal_type_field
        self.value_his_field = value_his_field
        self.train_res_field = train_res_field
        self.detection_config_field = detection_config_field
        self.day_start = day_start  # 几天以上开始天周期的检测
        self.week_start = week_start  # 几天以上开始周周期的检测
        self.agg = agg  # 检测窗口，单位分钟
        self.window = window  #
        self.sparse_period_thresh = sparse_period_thresh
        self.continuous_not_period_thresh = continuous_not_period_thresh
        self.his_window = his_window

        self.list_len_one_hour = int(60 / self.agg)  # 一小时有多少个点
        self.model = dict()  # 当前点的模型
        self.recent_sparse_period = []  # 最近sparse_period_thresh时间内稀疏周期突降异常的时间

        self.is_alg_upper_anomaly = False
        self.is_alg_lower_anomaly = False
        self.is_thresh_upper_anomaly = False
        self.is_thresh_lower_anomaly = False
        self.is_change_upper_anomaly = False
        self.is_change_lower_anomaly = False
        self.anomaly_type = ''

    # 稀疏周期型计算均值、标准差
    def cal_statistic_sparse_period(self, value_his, per_cor, is_cal_std):
        v_lis_list = []
        trans_window = self.list_len_one_hour * self.window // 2
        for i in range(1, len(value_his) // per_cor + 1):
            if i <= 4:
                try:
                    v_lis = [j for j in value_his[-per_cor * i - trans_window:-per_cor * i + trans_window] if j > 0]
                except:
                    v_lis = [j for j in value_his[:-per_cor * i + trans_window] if j > 0]
                v_lis = v_lis if v_lis else [0]
                v_lis_list.append(v_lis)
        mean_v = np.median([sum(i)/len(i) for i in v_lis_list])
        if is_cal_std:
            std_v = np.median([np.std(i) for i in v_lis_list])
            return mean_v, std_v
        return mean_v, None

    # 稀疏非周期型计算均值、标准差
    def cal_statistic_sparse_not_period(self, value_his, is_cal_std):
        dat = [i for i in value_his if i > 0]
        dat = dat if dat else [0]
        mean_v = sum(dat)/len(dat)
        if is_cal_std:
            std_v = np.std(dat)
            return mean_v, std_v
        return mean_v, None

    # 连续(非稀疏)周期型计算均值、标准差
    def cal_statistic_not_sparse_period(self, value_his, per_coef, detect_mode, is_cal_std):
        trans_window = self.list_len_one_hour * self.window // 2
        length = math.ceil(per_coef * trans_window)
        v_lis_list = []
        if detect_mode == 'day':
            v_lis_list.append(value_his[-self.list_len_one_hour * 24 - length:-self.list_len_one_hour * 24 + length + 1])
            v_lis_list.append(value_his[-self.list_len_one_hour * 48 - length:-self.list_len_one_hour * 48 + length + 1])
            days = len(value_his) // (self.list_len_one_hour * 24)
            if 3 <= days < 4:
                v_lis_list.append(value_his[-self.list_len_one_hour * 72 - length:-self.list_len_one_hour * 72 + length + 1])
            elif days >= 4:
                v_lis_list.append(value_his[-self.list_len_one_hour * 72 - length:-self.list_len_one_hour * 72 + length + 1])
                v_lis_list.append(value_his[-self.list_len_one_hour * 96 - length:-self.list_len_one_hour * 96 + length + 1])
        else:
            v_lis_list.append(value_his[-self.list_len_one_hour * 24 - length:-self.list_len_one_hour * 24 + length + 1])
            v_lis_list.append(value_his[-self.list_len_one_hour * 48 - length:-self.list_len_one_hour * 48 + length + 1])
            v_lis_list.append(value_his[-self.list_len_one_hour * 168 - length:-self.list_len_one_hour * 168 + length + 1])
            v_lis_list.append(value_his[-self.list_len_one_hour * 336 - length:-self.list_len_one_hour * 336 + length + 1])
        mean = np.median([sum(i)/len(i) for i in v_lis_list])
        if is_cal_std:
            std = np.median([np.std(i) for i in v_lis_list])
            return mean, std
        return mean, None

    # 连续(非稀疏)非周期型计算均值、标准差
    def cal_statistic_not_sparse_not_period(self, value_his, is_cal_std):
        value_his_ = value_his[-self.list_len_one_hour * 24 * self.continuous_not_period_thresh:]
        dat = [i for i in value_his_ if i > 0]
        dat = dat if dat else [0]
        mean_v = sum(dat)/len(dat)
        if is_cal_std:
            std_v = np.std(dat)
            return mean_v, std_v
        return mean_v, None

    # 计算均值、标准差
    def cal_statistic(self, value_his, is_sparse, is_period, per_cor, per_coef, detect_mode, is_cal_std):
        if is_sparse:  # 稀疏
            if is_period:  # 周期
                mean_v, std_v = self.cal_statistic_sparse_period(value_his, per_cor, is_cal_std)
            else:  # 非周期
                mean_v, std_v = self.cal_statistic_sparse_not_period(value_his, is_cal_std)
        else:  # 连续
            if is_period:  # 周期
                mean_v, std_v = self.cal_statistic_not_sparse_period(value_his, per_coef, detect_mode, is_cal_std)
            else:  # 非周期
                mean_v, std_v = self.cal_statistic_not_sparse_not_period(value_his, is_cal_std)
        return mean_v, std_v

    # 稀疏周期型
    def detect_sparse_period(self, value, mean_v, std_v, up_alg_enable, up_sigma, down_alg_enable, down_sigma):
        if up_alg_enable:
            upper = mean_v + up_sigma * std_v
            if value > upper:
                self.is_alg_upper_anomaly = True
                self.anomaly_type = 'sparse_period_upper'
        if down_alg_enable:
            lower = mean_v - down_sigma * std_v
            if value < lower:
                self.is_alg_lower_anomaly = True
                self.anomaly_type = 'sparse_period_lower'

    # 稀疏非周期型
    def sparse_not_period(self, value, mean_v, std_v, up_alg_enable, up_sigma):
        if up_alg_enable:
            cv = std_v / mean_v
            upper = mean_v + up_sigma * (1 + cv) * std_v
            if value > upper:
                self.is_alg_upper_anomaly = True
                self.anomaly_type = 'sparse_not_period_upper'

    # 连续(非稀疏)周期型
    def detect_not_sparse_period(self, value, mean_v, std_v, up_alg_enable, up_sigma, down_alg_enable, down_sigma):
        if up_alg_enable:
            upper = mean_v + up_sigma * std_v
            if value > upper:
                self.is_alg_upper_anomaly = True
                self.anomaly_type = 'continuous_period_upper'
        if down_alg_enable:
            lower = mean_v - down_sigma * std_v
            if value < lower:
                self.is_alg_lower_anomaly = True
                self.anomaly_type = 'continuous_period_lower'

    # 连续(非稀疏)非周期型
    def detect_not_sparse_not_period(self, value, mean_v, std_v, up_alg_enable, up_sigma, down_alg_enable, down_sigma):
        cv = std_v / mean_v
        if up_alg_enable:
            upper = mean_v + up_sigma * (1 + cv) * std_v
            if value > upper:
                self.is_alg_upper_anomaly = True
                self.anomaly_type = 'continuous_not_period_upper'
        if down_alg_enable:
            lower = mean_v - down_sigma * (1 + cv) * std_v
            if value < lower:
                self.is_alg_lower_anomaly = True
                self.anomaly_type = 'continuous_not_period_lower'

    # 算法检测
    def alg_detect(self, value, value_his, is_sparse, is_period, per_cor, per_coef, detect_mode,
                   up_alg_enable, up_sigma, down_alg_enable, down_sigma):
        mean_v, std_v = self.cal_statistic(value_his, is_sparse, is_period, per_cor, per_coef, detect_mode, is_cal_std=True)
        if is_sparse:  # 稀疏
            if is_period:  # 周期
                self.detect_sparse_period(value, mean_v, std_v, up_alg_enable, up_sigma, down_alg_enable, down_sigma)
            else:   # 非周期
                self.sparse_not_period(value, mean_v, std_v, up_alg_enable, up_sigma)
        else:   # 连续
            if is_period:  # 周期
                self.detect_not_sparse_period(value, mean_v, std_v, up_alg_enable, up_sigma, down_alg_enable, down_sigma)
            else:   # 非周期
                self.detect_not_sparse_not_period(value, mean_v, std_v, up_alg_enable, up_sigma, down_alg_enable, down_sigma)
        return mean_v, std_v

    # 固定阈值检测
    def thresh_detect(self, value, up_thresh_enable, up_thresh, down_thresh_enable, down_thresh):
        if up_thresh_enable and value > up_thresh:
            self.is_thresh_upper_anomaly = True
        if down_thresh_enable and value < down_thresh:
            self.is_thresh_lower_anomaly = True

    # 变化量阈值检测
    def change_detect(self, value, value_his, up_change_enable, up_change_thresh, down_change_enable, down_change_thresh,
                      mean_v, is_sparse, is_period, per_cor, per_coef, detect_mode):
        if mean_v is None:
            mean_v, std_v = self.cal_statistic(value_his, is_sparse, is_period, per_cor, per_coef, detect_mode, is_cal_std=False)
        varis = value - mean_v
        if up_change_enable and varis > 0 and varis > up_change_thresh:  # 增长量超过突增的变化量阈值
            self.is_change_upper_anomaly = True
        elif down_change_enable and varis < 0 and abs(varis) > down_change_thresh:  # 下降量超过突降的变化量阈值
            self.is_change_lower_anomaly = True

    def run(self, data):
        timestamp = int(data[self.time_field])   # 毫秒
        value = int(data[self.value_field])
        value_his = data[self.value_his_field]   # 长度从0~288*15
        train_res = data[self.train_res_field]
        detection_config = data[self.detection_config_field]
        # 参数
        up_alg_enable = detection_config['up_alg_enable']
        up_thresh_enable = detection_config['up_thresh_enable']
        up_change_enable = detection_config['up_change_enable'] if 'up_change_enable' in detection_config else False
        up_sigma = detection_config['up_sigma']
        up_thresh = detection_config['up_thresh']
        up_change_thresh = detection_config['up_change_thresh'] if 'up_change_thresh' in detection_config else 0
        down_alg_enable = detection_config['down_alg_enable']
        down_thresh_enable = detection_config['down_thresh_enable']
        down_change_enable = detection_config['down_change_enable'] if 'down_change_enable' in detection_config else False
        down_sigma = detection_config['down_sigma']
        down_thresh = detection_config['down_thresh']
        down_change_thresh = detection_config['down_change_thresh'] if 'down_change_thresh' in detection_config else 0

        self.is_alg_upper_anomaly = False
        self.is_alg_lower_anomaly = False
        self.is_thresh_upper_anomaly = False
        self.is_thresh_lower_anomaly = False
        self.is_change_upper_anomaly = False
        self.is_change_lower_anomaly = False
        self.anomaly_type = ''

        # 【开启固定阈值检测】
        if up_thresh_enable or down_thresh_enable:
            self.thresh_detect(value, up_thresh_enable, up_thresh, down_thresh_enable, down_thresh)
        mean_v = None
        day_start_len = self.list_len_one_hour * 24 * self.day_start
        week_start_len = self.list_len_one_hour * 24 * self.week_start
        lens = len(value_his)
        if lens >= day_start_len:
            # 稀疏性、周期性
            is_sparse = train_res['is_sparse']  # 该模板数据是否稀疏
            is_period = train_res['is_period']  # 该模板数据是否周期
            per_cor = train_res['per_cor']  # 该模板周期长度
            per_coef = train_res['per_coef']  #
            # 模式：天/周
            if day_start_len <= lens < week_start_len:
                detect_mode = 'day'
            else:
                detect_mode = 'week'
            # 【开启算法检测】
            if up_alg_enable or down_alg_enable:
                # 异常检测
                mean_v, std_v = self.alg_detect(value, value_his, is_sparse, is_period, per_cor, per_coef, detect_mode,
                                                up_alg_enable, up_sigma, down_alg_enable, down_sigma)
                # 过滤非连续的稀疏周期突降异常
                if self.anomaly_type == 'sparse_period_lower':
                    self.recent_sparse_period.append(timestamp)
                    cri_time = self.sparse_period_thresh * 60000
                    self.recent_sparse_period = [i for i in self.recent_sparse_period if (timestamp - i) < cri_time]
                    sparse_period_thresh_his = value_his[-self.sparse_period_thresh // self.agg:]
                    if len(self.recent_sparse_period) != sum([1 for i in sparse_period_thresh_his if i > 0]):
                        self.is_alg_lower_anomaly = False
                        self.anomaly_type = ''
            # 【开启变化量阈值检测】
            if up_change_enable or down_change_enable:
                self.change_detect(value, value_his, up_change_enable, up_change_thresh, down_change_enable,
                                   down_change_thresh, mean_v, is_sparse, is_period, per_cor, per_coef, detect_mode)

        # 综合3种检测结果
        is_upper_anomaly, is_lower_anomaly = True, True
        if (up_alg_enable and not self.is_alg_upper_anomaly) or \
                (up_thresh_enable and not self.is_thresh_upper_anomaly) or \
                (up_change_enable and not self.is_change_upper_anomaly) or \
                ((not up_alg_enable) and (not up_thresh_enable) and (not up_change_enable)):
            is_upper_anomaly = False
        if (down_alg_enable and not self.is_alg_lower_anomaly) or \
                (down_thresh_enable and not self.is_thresh_lower_anomaly) or \
                (down_change_enable and not self.is_change_lower_anomaly) or \
                ((not down_alg_enable) and (not down_thresh_enable) and (not down_change_enable)):
            is_lower_anomaly = False
        # 异常类型
        if is_upper_anomaly:
            data['abnormal_type'] = 1
        elif is_lower_anomaly:
            data['abnormal_type'] = 2

        # 非突增突降
        if (not is_upper_anomaly) and (not is_lower_anomaly):
            # 判断是否偶发
            if train_res['is_accidental']:
                data['abnormal_type'] = 7

        data['result'] = {'is_upper_anomaly': is_upper_anomaly,
                          'is_lower_anomaly': is_lower_anomaly,
                          'anomaly_type': self.anomaly_type if (is_upper_anomaly or is_lower_anomaly) else ''}
        self.update_model()

        return data

    def update_model(self):  # 模型更新
        self.model['recent_sparse_period'] = self.recent_sparse_period

    def load_state(self, dict):
        model = dict["model"]
        self.recent_sparse_period = model.get("recent_sparse_period")
        self.model = model

    def save_state(self):
        return {
            "model": self.model
        }

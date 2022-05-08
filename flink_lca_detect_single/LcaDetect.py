import numpy as np
import pandas as pd
import math
import sys
import pickle
import os
from datetime import datetime, date


class LcaDetect(object):

    @staticmethod
    def from_map(conf_dict):
        id_field = conf_dict.get("id_field")
        time_field = conf_dict.get("time_field")
        value_field = conf_dict.get("value_field")
        train_res_field = conf_dict.get("train_res_field")
        day_start = conf_dict.get("day_start", 2)
        week_start = conf_dict.get("week_start", 14)
        agg = conf_dict.get("agg", 5)
        sigma = conf_dict.get("sigma", 3)
        window = conf_dict.get("window", 1)
        sparse_period_thresh = conf_dict.get("sparse_period_thresh", 60)
        continuous_not_period_thresh = conf_dict.get("continuous_not_period_thresh", 7)
        his_window = conf_dict.get("his_window", 15)
        is_thresh = conf_dict.get("is_thresh", False)
        is_varis_thresh = conf_dict.get("is_varis_thresh", False)
        thresh_dict = conf_dict.get("thresh_dict", {'upper': float('inf'), 'lower': float('-inf')})
        varis_thresh_dict = conf_dict.get("varis_thresh_dict", {'upper': float('inf'), 'lower': float('-inf')})

        lca_detect = LcaDetect(id_field, time_field, value_field, train_res_field, day_start=day_start,
                               week_start=week_start, agg=agg, sigma=sigma, window=window,
                               sparse_period_thresh=sparse_period_thresh,
                               continuous_not_period_thresh=continuous_not_period_thresh, his_window=his_window,
                               is_thresh=is_thresh, is_varis_thresh=is_varis_thresh, thresh_dict=thresh_dict,
                               varis_thresh_dict=varis_thresh_dict)

        return lca_detect

    def __init__(self, id_field, time_field, value_field, train_res_field, day_start=2, week_start=14, agg=5, sigma=3,
                 window=1, sparse_period_thresh=60, continuous_not_period_thresh=7, his_window=15, is_thresh=False,
                 is_varis_thresh=False, thresh_dict={'upper': float('inf'), 'lower': float('-inf')},
                 varis_thresh_dict={'upper': float('inf'), 'lower': float('-inf')}):
        self.id_field = id_field
        self.time_field = time_field
        self.value_field = value_field
        self.train_res_field = train_res_field
        self.day_start = day_start  # 几天以上开始天周期的检测
        self.week_start = week_start  # 几天以上开始周周期的检测
        self.agg = agg  # 检测窗口，单位分钟
        self.sigma = sigma
        self.window = window  #
        self.sparse_period_thresh = sparse_period_thresh
        self.continuous_not_period_thresh = continuous_not_period_thresh
        self.his_window = his_window
        self.start_time = None  # str
        self.last_time = None
        self.is_start_detect = False
        # self.last_data_day = None
        self.detect_mode = 'day'
        self.list_len_one_hour = int(60 / self.agg)  # 一小时有多少个点
        self.is_thresh = is_thresh  # 是否开启固定阈值检测
        self.is_varis_thresh = is_varis_thresh  # 是否开启变化量阈值检测
        self.thresh_dict = thresh_dict  # 固定阈值检测时突增突降阈值
        self.varis_thresh_dict = varis_thresh_dict  # 变化量阈值检测时突增突降变化量阈值

        self.value_his = []  # 存储该模板过去14天数据
        self.is_sparse = False  # 该模板数据是否稀疏
        self.is_period = False  # 该模板数据是否周期
        self.per_cor = None  # 该模板周期长度
        self.per_coef = None  #
        self.recent_sparse_period = []  # 最近sparse_period_thresh时间内稀疏周期突降异常的时间

        self.is_upper_anomaly = False
        self.is_lower_anomaly = False
        self.anomaly_type = ''

    # 稀疏周期型
    def detect_sparse_period(self, value):
        cor = self.per_cor
        sta_mean = []
        sta_std = []
        trans_window = self.list_len_one_hour * self.window // 2
        for i in range(1, len(self.value_his) // cor + 1):
            if i <= 4:
                try:
                    v_lis = [j for j in self.value_his[-cor * i - trans_window:-cor * i + trans_window] if j > 0]
                except:
                    v_lis = [j for j in self.value_his[:-cor * i + trans_window] if j > 0]
                v_lis = v_lis if v_lis else [0]
                sta_mean.append(np.mean(v_lis))
                sta_std.append(np.std(v_lis))
        mean_v = np.median(sta_mean)
        if self.is_varis_thresh:  # 开启变化量阈值检验
            varis = value - mean_v
            if varis < 0 and abs(varis) > self.varis_thresh_dict['lower']:  # 下降量超过突降的变化量阈值
                if (not self.is_thresh) or (self.is_thresh and value < self.thresh_dict['lower']):  # 固定阈值检测
                    std_v = np.median(sta_std)
                    lower = mean_v - self.sigma * std_v
                    if value < lower:
                        self.is_lower_anomaly = True
                        self.anomaly_type = 'sparse_period_lower'
            elif varis > 0 and varis > self.varis_thresh_dict['upper']:  # 增长量超过突增的变化量阈值
                if (not self.is_thresh) or (self.is_thresh and value > self.thresh_dict['upper']):  # 固定阈值检测
                    std_v = np.median(sta_std)
                    upper = mean_v + self.sigma * std_v
                    if value > upper:
                        self.is_upper_anomaly = True
                        self.anomaly_type = 'sparse_period_upper'
        else:
            if self.is_thresh and value < self.thresh_dict['lower']:  # 固定阈值检测
                std_v = np.median(sta_std)
                lower = mean_v - self.sigma * std_v
                if value < lower:
                    self.is_lower_anomaly = True
                    self.anomaly_type = 'sparse_period_lower'
            elif self.is_thresh and value > self.thresh_dict['upper']:  # 固定阈值检测
                std_v = np.median(sta_std)
                upper = mean_v + self.sigma * std_v
                if value > upper:
                    self.is_upper_anomaly = True
                    self.anomaly_type = 'sparse_period_upper'
            elif not self.is_thresh:
                std_v = np.median(sta_std)
                lower = mean_v - self.sigma * std_v
                upper = mean_v + self.sigma * std_v
                if value < lower:
                    self.is_lower_anomaly = True
                    self.anomaly_type = 'sparse_period_lower'
                if value > upper:
                    self.is_upper_anomaly = True
                    self.anomaly_type = 'sparse_period_upper'

    # 稀疏非周期型
    def sparse_not_period(self, value):
        dat = [i for i in self.value_his if i > 0]
        dat = dat if dat else [0]
        mean_v = np.mean(dat)
        if self.is_varis_thresh:
            varis = value - mean_v
            if varis > 0 and varis > self.varis_thresh_dict['upper']:  # 增长量超过突增的变化量阈值
                if (not self.is_thresh) or (self.is_thresh and value > self.thresh_dict['upper']):  # 固定阈值检测
                    std_v = np.std(dat)
                    cv = std_v / mean_v
                    upper = mean_v + self.sigma * (1 + cv) * std_v
                    if value > upper:
                        self.is_upper_anomaly = True
                        self.anomaly_type = 'sparse_not_period_upper'
        else:
            if (not self.is_thresh) or (self.is_thresh and value > self.thresh_dict['upper']):  # 固定阈值检测
                std_v = np.std(dat)
                cv = std_v / mean_v
                upper = mean_v + self.sigma * (1 + cv) * std_v
                if value > upper:
                    self.is_upper_anomaly = True
                    self.anomaly_type = 'sparse_not_period_upper'

    # 连续(非稀疏)周期型
    def detect_not_sparse_period(self, value):
        trans_window = self.list_len_one_hour * self.window // 2
        length = math.ceil(self.per_coef * trans_window)
        if self.detect_mode == 'day':
            mean1 = np.mean(
                self.value_his[
                -self.list_len_one_hour * 24 - length:-self.list_len_one_hour * 24 + length + 1])  # 。。。。。。。。。
            mean2 = np.mean(
                self.value_his[-self.list_len_one_hour * 48 - length:-self.list_len_one_hour * 48 + length + 1])
            std1 = np.std(
                self.value_his[-self.list_len_one_hour * 24 - length:-self.list_len_one_hour * 24 + length + 1])
            std2 = np.std(
                self.value_his[-self.list_len_one_hour * 48 - length:-self.list_len_one_hour * 48 + length + 1])
            days = len(self.value_his) // (self.list_len_one_hour * 24)
            if 2 <= days < 3:
                mean = np.median([mean1, mean2])
                std = np.median([std1, std2])
            elif 3 <= days < 4:
                mean3 = np.mean(self.value_his[
                                -self.list_len_one_hour * 72 - length:-self.list_len_one_hour * 72 + length + 1])
                std3 = np.std(self.value_his[
                              -self.list_len_one_hour * 72 - length:-self.list_len_one_hour * 72 + length + 1])
                mean = np.median([mean1, mean2, mean3])
                std = np.median([std1, std2, std3])
            else:
                mean3 = np.mean(self.value_his[
                                -self.list_len_one_hour * 72 - length:-self.list_len_one_hour * 72 + length + 1])
                std3 = np.std(self.value_his[
                              -self.list_len_one_hour * 72 - length:-self.list_len_one_hour * 72 + length + 1])
                mean4 = np.mean(self.value_his[
                                -self.list_len_one_hour * 96 - length:-self.list_len_one_hour * 96 + length + 1])
                std4 = np.std(self.value_his[
                              -self.list_len_one_hour * 96 - length:-self.list_len_one_hour * 96 + length + 1])
                mean = np.median([mean1, mean2, mean3, mean4])
                std = np.median([std1, std2, std3, std4])
        else:
            mean1 = np.mean(
                self.value_his[-self.list_len_one_hour * 24 - length:-self.list_len_one_hour * 24 + length + 1])
            mean2 = np.mean(
                self.value_his[-self.list_len_one_hour * 48 - length:-self.list_len_one_hour * 48 + length + 1])
            mean3 = np.mean(self.value_his[
                            -self.list_len_one_hour * 168 - length:-self.list_len_one_hour * 168 + length + 1])
            mean4 = np.mean(self.value_his[
                            -self.list_len_one_hour * 336 - length:-self.list_len_one_hour * 336 + length + 1])
            mean = np.median([mean1, mean2, mean3, mean4])
            std1 = np.std(
                self.value_his[-self.list_len_one_hour * 24 - length:-self.list_len_one_hour * 24 + length + 1])
            std2 = np.std(
                self.value_his[-self.list_len_one_hour * 48 - length:-self.list_len_one_hour * 48 + length + 1])
            std3 = np.std(self.value_his[
                          -self.list_len_one_hour * 168 - length:-self.list_len_one_hour * 168 + length + 1])
            std4 = np.std(self.value_his[
                          -self.list_len_one_hour * 336 - length:-self.list_len_one_hour * 336 + length + 1])
            std = np.median([std1, std2, std3, std4])
        if self.is_varis_thresh:  # 开启变化量阈值检验
            varis = value - mean
            if varis < 0 and abs(varis) > self.varis_thresh_dict['lower']:  # 下降量超过突降的变化量阈值
                if (not self.is_thresh) or (self.is_thresh and value < self.thresh_dict['lower']):  # 固定阈值检测
                    lower = mean - self.sigma * std
                    if value < lower:
                        self.is_lower_anomaly = True
                        self.anomaly_type = 'continuous_period_lower'
            elif varis > 0 and varis > self.varis_thresh_dict['upper']:  # 增长量超过突增的变化量阈值
                if (not self.is_thresh) or (self.is_thresh and value > self.thresh_dict['upper']):  # 固定阈值检测
                    upper = mean + self.sigma * std
                    if value > upper:
                        self.is_upper_anomaly = True
                        self.anomaly_type = 'continuous_period_upper'
        else:
            if self.is_thresh and value > self.thresh_dict['upper']:  # 固定阈值检测
                upper = mean + self.sigma * std
                if value > upper:
                    self.is_upper_anomaly = True
                    self.anomaly_type = 'continuous_period_upper'
            elif self.is_thresh and value < self.thresh_dict['lower']:  # 固定阈值检测
                lower = mean - self.sigma * std
                if value < lower:
                    self.is_lower_anomaly = True
                    self.anomaly_type = 'continuous_period_lower'
            if not self.is_thresh:
                upper = mean + self.sigma * std
                lower = mean - self.sigma * std
                if value > upper:
                    self.is_upper_anomaly = True
                    self.anomaly_type = 'continuous_period_upper'
                if value < lower:
                    self.is_lower_anomaly = True
                    self.anomaly_type = 'continuous_period_lower'

    # 连续(非稀疏)非周期型
    def detect_not_sparse_not_period(self, value):
        dat = [i for i in self.value_his[-self.list_len_one_hour * 24 * self.continuous_not_period_thresh:] if
               i > 0]
        dat = dat if dat else [0]
        mean_v = np.mean(dat)
        if self.is_varis_thresh:  # 开启变化量阈值检验
            varis = value - mean_v
            if varis < 0 and abs(varis) > self.varis_thresh_dict['lower']:  # 下降量超过突降的变化量阈值
                if (not self.is_thresh) or (self.is_thresh and value < self.thresh_dict['lower']):  # 固定阈值检测
                    std_v = np.std(dat)
                    cv = std_v / mean_v
                    lower = mean_v - self.sigma * (1 + cv) * std_v
                    if value < lower:
                        self.is_lower_anomaly = True
                        self.anomaly_type = 'continuous_not_period_lower'
            elif varis > 0 and varis > self.varis_thresh_dict['upper']:  # 增长量超过突增的变化量阈值
                if (not self.is_thresh) or (self.is_thresh and value > self.thresh_dict['upper']):  # 固定阈值检测
                    std_v = np.std(dat)
                    cv = std_v / mean_v
                    upper = mean_v + self.sigma * (1 + cv) * std_v
                    if value > upper:
                        self.is_upper_anomaly = True
                        self.anomaly_type = 'continuous_not_period_upper'
        else:
            if self.is_thresh and value > self.thresh_dict['upper']:  # 固定阈值检测
                std_v = np.std(dat)
                cv = std_v / mean_v
                upper = mean_v + self.sigma * (1 + cv) * std_v
                if value > upper:
                    self.is_upper_anomaly = True
                    self.anomaly_type = 'continuous_not_period_upper'
            elif self.is_thresh and value < self.thresh_dict['lower']:  # 固定阈值检测
                std_v = np.std(dat)
                cv = std_v / mean_v
                lower = mean_v - self.sigma * (1 + cv) * std_v
                if value < lower:
                    self.is_lower_anomaly = True
                    self.anomaly_type = 'continuous_not_period_lower'
            elif not self.is_thresh:
                std_v = np.std(dat)
                cv = std_v / mean_v
                upper = mean_v + self.sigma * (1 + cv) * std_v
                lower = mean_v - self.sigma * (1 + cv) * std_v
                if value > upper:
                    self.is_upper_anomaly = True
                    self.anomaly_type = 'continuous_not_period_upper'
                if value < lower:
                    self.is_lower_anomaly = True
                    self.anomaly_type = 'continuous_not_period_lower'

    def detect(self, value):
        self.is_upper_anomaly = False
        self.is_lower_anomaly = False
        self.anomaly_type = ''
        # 固定阈值检测
        if self.is_thresh and self.thresh_dict['lower'] < value < self.thresh_dict['upper']:
            sys.exit()
        # 稀疏
        if self.is_sparse:
            # 周期
            if self.is_period:
                self.detect_sparse_period(value)
            # 非周期
            else:
                self.sparse_not_period(value)
        # 连续
        else:
            if self.is_period:
                self.detect_not_sparse_period(value)
            else:
                self.detect_not_sparse_not_period(value)

    def run(self, data):
        timestamp = int(data[self.time_field])   # 毫秒
        value = int(data[self.value_field])
        template_id = str(data[self.id_field])
        train_res = data[self.train_res_field]

        # 读取稀疏性、周期性
        self.is_sparse = train_res[template_id]['is_sparse']  # 该模板数据是否稀疏
        self.is_period = train_res[template_id]['is_period']  # 该模板数据是否周期
        self.per_cor = train_res[template_id]['per_cor']  # 该模板周期长度
        self.per_coef = train_res[template_id]['per_coef']  #

        if not self.start_time:
            self.start_time = timestamp
        else:
            millseconds = timestamp - self.start_time
            day_start_sec = self.day_start * 86400000
            week_start_sec = self.week_start * 86400000
            if not self.is_start_detect:
                # 2天以上历史数据才检测
                if millseconds >= day_start_sec:
                    self.is_start_detect = True
            if day_start_sec <= millseconds < week_start_sec:
                self.detect_mode = 'day'
            elif millseconds >= week_start_sec:
                self.detect_mode = 'week'

        # 保存最多最近 his_window天 数据
        if self.last_time:
            self.value_his += [0] * (int((timestamp - self.last_time) // (self.agg * 60000)) - 1)
        self.value_his = self.value_his[-min(self.list_len_one_hour * 24 * self.his_window, len(self.value_his)):]

        # 异常检测
        if self.is_start_detect:
            self.detect(value)
            # 过滤非连续的稀疏周期突降异常
            if self.anomaly_type == 'sparse_period_lower':
                self.recent_sparse_period.append(timestamp)
                cri_time = self.sparse_period_thresh * 60000
                self.recent_sparse_period = [i for i in self.recent_sparse_period if (timestamp - i) < cri_time]
                sparse_period_thresh_his = self.value_his[-self.sparse_period_thresh // self.agg:]
                if len(self.recent_sparse_period) != sum([1 for i in sparse_period_thresh_his if i > 0]):
                    self.is_lower_anomaly = False
                    self.anomaly_type = ''

        self.value_his.append(value)
        self.last_time = timestamp

        data['result'] = {'is_upper_anomaly': self.is_upper_anomaly,
                          'is_lower_anomaly': self.is_lower_anomaly,
                          'anomaly_type': self.anomaly_type}
        return data
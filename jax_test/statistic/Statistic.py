from jax_python.rad import RealtimeAnomalyDetect
import numpy as np
import time


class Statistic(RealtimeAnomalyDetect):

    def __init__(self):
        super(Statistic,self).__init__()
        self.config={}

    def configure(self, dict):
        self.config=dict
        self.Sigma = Sigma.init_sigma(dict)

    def score(self, record, key=None):
        print("key is:%s"%str(key))
        if key is not None:
            if self.contains_key(key) and self.state[key] is not None:
                self.Sigma.load_state(self.state[key])
            else:
                self.Sigma=Sigma.init_sigma(self.config)
        result=self.Sigma.run(record)
        if key is not None:
            self.state[key]=self.Sigma.save_state()
        return result


class Sigma(object):

    def __init__(self,time_field,value_field,granularity=60,increase_sigma=3.0,drop_sigma=3.0,window=120,
                 max_value=-1.0,min_value=-1.0,compensate_mode='negative',compensate_coefficient=0.1,
                 seasonal_anomaly_condition='day,7,3',seasonal_anomaly_threshold=0.75,smooth_window=30,
                 upper_constant=-1.0,lower_constant=-1.0,upper_detection=True,lower_detection=True):
        self.time_field=time_field
        self.value_field=value_field
        self.granularity=granularity
        self.length_per_day=int(60*24*60/self.granularity)

        self.increase_sigma=increase_sigma
        self.drop_sigma=drop_sigma
        self.window=window
        self.smooth_window=smooth_window
        self.max_value=max_value
        self.min_value=min_value
        self.compensate_mode=compensate_mode
        self.compensate_coefficient=compensate_coefficient
        self.seasonal_anomaly_condition=seasonal_anomaly_condition
        self.seasonal_type = self.seasonal_anomaly_condition.split(',')[0]
        self.check_season = int(self.seasonal_anomaly_condition.split(',')[1])
        self.check_season = max(1, self.check_season)
        self.occur_season = int(self.seasonal_anomaly_condition.split(',')[2])
        self.occur_season = max(1, self.occur_season)
        self.seasonal_anomaly_threshold = seasonal_anomaly_threshold
        self.upper_constant = upper_constant
        self.lower_constant = lower_constant
        self.upper_detection=upper_detection
        self.lower_detection=lower_detection
        self.cursor=0
        self.statistic_data=np.zeros(self.window)
        self.anomaly_date=[]
        self.upper_array=np.zeros(self.smooth_window)
        self.lower_array=np.zeros(self.smooth_window)
        self.std=0.0
        self.pre_value=0.0
        self.anomaly=0
        self.abnormality=0
        self.upper=0.0
        self.lower=0.0

    def baseline_compensation(self):
        '''
        基带补偿
        :return:
        '''
        if self.min_value < 0 or self.max_value <= 0 or self.min_value >= self.max_value:
            compensate = 0.0
            base =self.std
        else:
            if self.pre_value > self.max_value:
                nor_fit = 1
            elif self.pre_value < self.min_value:
                nor_fit = 0
            else:
                nor_fit = (self.pre_value - self.min_value) / (self.max_value - self.min_value)
            if self.compensate_mode == 'negative':
                compensate_degree = 1 - nor_fit
                compensate_base = max((self.max_value - self.pre_value),0.0) * self.compensate_coefficient * compensate_degree
                compensate = compensate_degree * compensate_base
                base = self.std * compensate_degree
            elif self.compensate_mode == 'positive':
                compensate_degree = nor_fit
                compensate_base = max((self.max_value - self.pre_value),0.0) * self.compensate_coefficient * compensate_degree
                compensate = compensate_degree * compensate_base
                base = self.std * compensate_degree
            elif self.compensate_mode == 'both':
                if nor_fit == 0.5:
                    compensate_degree = 1
                elif nor_fit > 0.5:
                    compensate_degree = 1 - nor_fit
                else:
                    compensate_degree = nor_fit
                compensate_base = max((self.max_value - self.pre_value),0.0) * self.compensate_coefficient * compensate_degree
                compensate = compensate_degree * compensate_base
                base = self.std * compensate_degree
            else:
                compensate = 0.0
                base = self.std
        return base, compensate

    def get_baseline(self):
        self.pre_value = sum(self.statistic_data)/len(self.statistic_data)
        self.std = np.std(self.statistic_data)
        base,compensate=self.baseline_compensation()
        if self.cursor<self.window:
            self.upper=0.0
            self.lower=0.0
        else:
            self.upper=self.pre_value+self.increase_sigma*base+compensate
            self.lower=self.pre_value-self.drop_sigma*base-compensate
            if self.upper_constant!=-1.0 and self.upper>self.upper_constant:
                self.upper=self.upper_constant
            if self.lower_constant!=-1.0 and self.lower<self.lower_constant:
                self.lower=self.lower_constant
        self.upper_array = np.append(self.upper_array, self.upper)[-self.smooth_window:]
        self.lower_array = np.append(self.lower_array, self.lower)[-self.smooth_window:]
        self.upper = np.max(self.upper_array)
        self.lower = np.min(self.lower_array)

    def seasonal_anomaly_drop(self):
        '''
        周期异常消除
        :return:
        '''
        if self.min_value >= 0 and self.max_value > 0 and self.min_value < self.max_value:
            if self.x > self.max_value * self.seasonal_anomaly_threshold or self.x < self.min_value * (1 - self.seasonal_anomaly_threshold):
                drop_condition = False
            else:
                drop_condition = True
        else:
            drop_condition = True
        if self.anomaly == 1:
            if not drop_condition:
                pass
            else:
                current_seconds = time.mktime(time.strptime(self.t, '%Y-%m-%d %H:%M:%S'))
                delta_date = []
                delta_day = []
                for i in range(len(self.anomaly_date)):
                    date = self.anomaly_date[i]
                    if date != '':
                        seconds = time.mktime(time.strptime(self.anomaly_date[i], '%Y-%m-%d %H:%M:%S'))
                        if self.seasonal_type == 'week':
                            for j in range(min(4, self.check_season)):
                                if abs(current_seconds - seconds) <= 86400 * (j + 1) * 7 + self.smooth_window * 60 and abs(current_seconds - seconds) >= 86400 * (j + 1) * 7 - self.smooth_window * 60:
                                    delta_date.append(date)
                                    delta_day.append(date[:10])
                        else:
                            for j in range(min(28, self.check_season)):
                                if abs(current_seconds - seconds) <= 86400 * (j + 1) + self.smooth_window * 60 and abs(current_seconds - seconds) >= 86400 * (j + 1) - self.smooth_window * 60:
                                    delta_date.append(date)
                                    delta_day.append(date[:10])
                delta_day = list(set(delta_day))
                if len(delta_day) >= self.occur_season:
                    self.anomaly = 0
                    if self.x - self.pre_value > 0:
                        self.upper = self.x * (1.05)
                        self.upper_array[-1] = self.upper
                        self.upper = np.max(self.upper_array)
                    else:
                        self.lower = self.x * (0.95)
                        self.lower_array[-1] = self.lower
                        self.lower = np.min(self.lower_array)
            if len(self.anomaly_date)>self.length_per_day*28:
                self.anomaly_date.append(self.t)
                self.anomaly_date=self.anomaly_date[-self.length_per_day*28:]
            else:
                self.anomaly_date.append(self.t)

    def anomaly_detection(self):
        if self.cursor<self.window:
            self.anomaly=0
            self.abnormality=0
            self.pre_value=0.0
            self.upper=0.0
            self.lower=0.0
        else:
            if self.x>self.upper or self.x<self.lower:
                self.anomaly=1
                self.abnormality=int()
            else:
                self.anomaly=0
            if self.x > self.upper and not self.upper_detection:
                self.anomaly = 0
            if self.x < self.lower and not self.lower_detection:
                self.anomaly = 0
            if self.seasonal_type!='no':
               self.seasonal_anomaly_drop()
            if self.upper_constant != -1.0 and self.upper > self.upper_constant:
                self.upper = self.upper_constant
                if self.x > self.upper and self.upper_detection:
                    self.anomaly = 1
            if self.lower_constant != -1.0 and self.lower < self.lower_constant:
                self.lower = self.lower_constant
                if self.x < self.lower and self.lower_detection:
                    self.anomaly = 1
            if self.anomaly==0:
                self.abnormality=0
            else:
                zscore=(self.x-self.pre_value)/self.std if self.std!=0 else 0
                self.abnormality=int(abs(zscore)*10)
                if self.abnormality>=100:
                    self.abnormality=99
                if self.abnormality>0 and self.abnormality<1:
                    self.abnormality=1

    def run(self,data):
        self.t=data[self.time_field]
        self.x=data[self.value_field]
        self.get_baseline()
        self.anomaly_detection()
        self.statistic_data=np.append(self.statistic_data,self.x)
        self.cursor=self.cursor+1
        result=dict(timestamp=self.t,value=self.x,pre_value=self.pre_value,upper=self.upper,
                    lower=self.lower,anomaly=self.anomaly,abnormality=self.abnormality)
        return result

    def load_state(self,dict):

        self.cursor=dict.get("cursor")
        self.anomaly_date=list(dict.get("anomaly_date"))
        self.statistic_data=np.array(dict.get("statistic_data"))
        self.upper_array=np.array(dict.get("upper_array"))
        self.lower_array=np.array(dict.get("lower_array"))

    def save_state(self):
        model=dict(cursor=self.cursor,
                   anomaly_date=self.anomaly_date,
                   statistic_data=self.statistic_data.tolist(),
                   upper_array=self.upper_array.tolist(),
                   lower_array=self.lower_array.tolist())
        return model

    @staticmethod
    def init_sigma(config):
        time_field=config.get('time_field')
        value_field=config.get('value_field')
        granularity=config.get('granularity',60)
        increase_sigma=config.get('increase_sigma',3.0)
        drop_sigma=config.get('drop_sigma',3.0)
        window=config.get('window',120)
        max_value=config.get('max_value',-1.0)
        min_value=config.get('min_value',-1.0)
        compensate_mode=config.get('compensate_mode','negative')
        compensate_coefficient=config.get('compensate_coefficient',0.1)
        seasonal_anomaly_condition = config.get("seasonal_anomaly_condition", "day,7,3")
        seasonal_anomaly_threshold = config.get("seasonal_anomaly_threshold", 0.75)
        smooth_window = config.get("smooth_window", 30)
        upper_constant = config.get("upper_constant", -1.0)
        lower_constant = config.get("lower_constant", -1.0)
        upper_detection = config.get("upper_detection", True)
        lower_detection = config.get("lower_detection", True)
        sigmaALg = Sigma(time_field=time_field,
                         value_field=value_field,
                         granularity=granularity,
                         increase_sigma=increase_sigma,
                         drop_sigma=drop_sigma,
                         window=window,
                         max_value=max_value,
                         min_value=min_value,
                         compensate_mode=compensate_mode,
                         compensate_coefficient=compensate_coefficient,
                         seasonal_anomaly_condition=seasonal_anomaly_condition,
                         seasonal_anomaly_threshold=seasonal_anomaly_threshold,
                         smooth_window=smooth_window,
                         upper_constant=upper_constant,
                         lower_constant=lower_constant,
                         upper_detection=upper_detection,
                         lower_detection=lower_detection)
        return sigmaALg
import math
import random
import time

class Mock(object):

    def __init__(self) -> None:
        self.index = 1
        self.i3 = random.random() * math.pow(2, 3)
        self.i4 = random.random() * math.pow(2, 4)
        self.i5 = random.random() * math.pow(2, 5)
        self.i6 = random.random() * math.pow(2, 6)
        self.i7 = random.random() * math.pow(2, 7)
        self.periodic_width = 10000
        self.begin_ts = int(time.mktime(time.strptime('2020-01-01', "%Y-%m-%d")))
        self.current_ts = self.begin_ts
        super().__init__()



    def gen_point(self, t):
        sum_point = 0
        for i in range(3, 8):
            x = {
                3: self.i3,
                4: self.i4,
                5: self.i5,
                6: self.i6,
                7: self.i7
            }

            sum_point = sum_point + 1 / math.pow(2, i) * math.sin(2 * math.pi * t * (math.pow(2, 2 + i) + x[i]))
        return sum_point

    def gen_ordered_data(self,width=1):
        current_time = time.localtime(self.current_ts)
        self.current_ts = self.current_ts + 60
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", current_time)

        if self.index * width > self.periodic_width:
            self.index = 1

        value = self.gen_point( self.index * width / self.periodic_width)
        self.index = self.index + 1

        metrics = ['cpu','gpu','disk','memory']
        metric = metrics[random.randint(0,3)]

        return {
            "time":current_time_str,
            "value":value,
            "metric":metric
        }



    def gen_data(self, width = 1):
        ran_ts = random.randint(self.begin_ts, self.begin_ts + 365 * 24 * 60 * 60 * 2)
        ran_time = time.localtime(ran_ts)
        ran_time_str = time.strftime("%Y-%m-%d %H:%M:%S", ran_time)

        if self.index * width > self.periodic_width:
            self.index = 1

        value = self.gen_point( self.index * width / self.periodic_width)
        self.index = self.index + 1

        metrics = ['cpu','gpu','disk','memory']
        metric = metrics[random.randint(0,3)]

        return {
            "time":ran_time_str,
            "value":value,
            "metric":metric
        }

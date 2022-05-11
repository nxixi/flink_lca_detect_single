import random
from jax_python import rad

class MovingAvg(rad.RealtimeAnomalyDetect):

    def __init__(self):
        super(MovingAvg, self).__init__()
        self.config = {}

    def configure(self, dict):
        self.config = dict

    def score(self, record, key=None):
        # if random.random() > 0.8:
        #     raise Exception("simulate error")
        value_field = self.config['valueField']
        value = int(record[value_field])
        count = 1
        sum = value
        if key is not None:
            if self.contains_key(key) and self.state[key] is not None:
                count = self.state[key]["count"] + 1
                sum = self.state[key]["sum"] + value
                self.state[key]["count"] = count
                self.state[key]["sum"] = sum
            else:
                self.state[key] = {
                    "count": count,
                    "sum": sum
                }
        std = int(self.config['std'] if self.config['std'] is not None else 0)
        if (sum / count) + std < value or value < (sum / count) - std:
            record['anomaly'] = 1.0
        else:
            record['anomaly'] = 0.0
        return record

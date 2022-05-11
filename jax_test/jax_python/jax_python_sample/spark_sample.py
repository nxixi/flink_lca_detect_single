from jax_python import aad
import json

class TotalAvg(aad.AbstractAbnormalDetect):

    def __init__(self):
        self.config = {}

    def configure(self, dict):
        self.config = dict

    def fields(self):
        return [('anomaly', 'float')]

    def transform(self, *df):
        import pandas as pd
        _df = df[0]
        value_field = self.config['valueField']
        std = self.config['std']
        mean = _df[value_field].mean()
        _df['anomaly'] = [1.0 if abs(x-mean) > std else 0.0 for x in _df[value_field]]
        model = {'mean': mean}
        return _df, bytes(json.dumps(model), encoding='utf-8')


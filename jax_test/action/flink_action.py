import time

from jax_python.rad import RealtimeAnomalyDetect
from ..source.mock import Mock
from ..watcher.default_watcher import DefaultWatcher
from ..watcher.watcher import Watcher
import pandas as pd

severity2level = {'警告': 2, '次要': 3, '重要': 4, '紧急': 5}

class Transporter(object):

    def __init__(self) -> None:
        default_watcher = DefaultWatcher()
        self.watchers = list()
        self.watchers.append(default_watcher)
        self.mock = Mock()
        super().__init__()

    def add_watcher(self, watcher: Watcher):
        self.watchers.append(watcher)

    def transport(self, conf: dict, rad: RealtimeAnomalyDetect, time_sorted=True):
        rad.configure(conf)
        watchers = self.watchers

        # while True:
        #     if time_sorted:
        #         record = self.mock.gen_ordered_data()
        #     else:
        #         record = self.mock.gen_data()
        #
        #     result = rad.score(record, record['metric'])
        #
        #     for watcher in watchers:
        #         watcher.action(result)
        #
        #     time.sleep(1)

        # a = pd.read_csv('after_result12.csv')
        # b = pd.read_csv('after_result13.csv')
        # #
        # # error_time = list(a[a['hits'] != b['hits']]['timestamp'])
        # #
        # print(False in list(a['ents'] == b['ents']), False in list(a['hits'] == b['hits']), False in list(a['recommendLevels'] == b['recommendLevels']))

        # xiamen_data = pd.read_csv('/Users/apple/PycharmProjects/refiner-ai/calEntropy/xiamen_content_0.5.csv', usecols=[2, 5], names=['first_layer_template_id', 'timestamp'], header=0)
        # xiamen_data['timestamp'] = pd.to_datetime(xiamen_data['timestamp'])
        # xiamen_data['time'] = xiamen_data['timestamp']
        # xiamen_data['month'] = xiamen_data['timestamp'].apply(lambda x: x.month)
        # xiamen_data = xiamen_data[xiamen_data['month'] >= 8]
        # xiamen_data['day'] = xiamen_data['timestamp'].apply(lambda x: x.date())
        # xiamen_data['hour'] = xiamen_data['timestamp'].apply(lambda x: x.hour)
        # xiamen_data = xiamen_data.sort_values(by='timestamp')
        # xiamen_data = xiamen_data.reset_index(drop=True)
        # xiamen_data['timestamp'] = xiamen_data['timestamp'].apply(
        #     lambda x: int(time.mktime(time.strptime(str(x), "%Y-%m-%d %H:%M:%S"))))
        # xiamen_data['timestamp'] = xiamen_data['timestamp'].apply(lambda x: x * 1000)

        # xiamen_data.to_csv('xiamen_processed_data.csv', index=False)


        # data = pd.read_csv('test_data/xiamen_processed_data.csv')
        # data['severity'] = data['重要程度'].apply(lambda x: severity2level[x])
        #
        def time2stamp(tim):
            timeArray = time.strptime(str(tim), "%Y-%m-%d %H:%M:%S")
            timeStamp = int(time.mktime(timeArray))
            return timeStamp*1000
        #
        # data = pd.read_csv('test_data/entropy_test1.csv')
        # data['timestamp'] = pd.to_datetime(data['timestamp']).apply(time2stamp)

        from collections import Counter
        # data = pd.read_csv('/Users/apple/Downloads/模板-时间.csv')
        # data = data.rename(columns={'tid': 'template_id'})
        # data = pd.read_csv('/Users/apple/Downloads/工单分组(1).csv')
        # data = data.rename(columns={'对象+模板': 'template_id', '告警派单时间': 'time'})
        # count = Counter(data['template_id'])
        # id_list = [key for key in count if count[key] > 2]
        # data = data[data['template_id'].isin(id_list)]
        # # for i in id_list:
        # #     data = data.append(pd.DataFrame({'time': ['2021-05-27 00:00:00'], 'template_id': [i]}))
        # for i in id_list:
        #     data = data.append(pd.DataFrame({'time': ['2021-06-11T00:00:00+08:00'], 'template_id': [i]}))
        # data['timestamp'] = pd.to_datetime(data['time'].apply(lambda x: x[:-6])).apply(time2stamp)
        # data['severity'] = 1
        # data = data[~data['template_id'].isna()]
        # tem2id = dict(zip(sorted(set(data['template_id'])), range(len(set(data['template_id'])))))
        # # tem2id_df = pd.DataFrame({'tem': list(tem2id.keys()), 'id': list(tem2id.values())})
        # # tem2id_df.to_csv('res1/tem2id.csv', index=False)
        # data['template_id'] = data['template_id'].apply(lambda x: tem2id[x])


        data_ori = pd.read_excel('/Users/apple/Desktop/上海银行2019_01-03去重.xlsx')
        data = data_ori[['alm_occur_time', 'nn', 'templateId']]
        data['template_id'] = data['nn']+data['templateId'].apply(lambda x: '-' + str(x))
        data['timestamp'] = data['alm_occur_time'].apply(lambda x: time2stamp(x.split('+')[0]))
        data['severity'] = 1

        #############

        s0 = time.time()
        max_time = 0
        ents = []
        recommendLevels = []
        hits = []
        long_length = 0

        for i in range(len(data)):
            tid = data['template_id'].iloc[i]
            tim = data['timestamp'].iloc[i]
            til = data['severity'].iloc[i]

            record = {'template_id': tid, 'timestamp': tim, "level": til}

            s = time.time()
            result = rad.score(record)
            e = time.time()
            print(e-s)
            if e-s > 0.09:
                long_length += 1

            ents.append(result['entropy'])
            recommendLevels.append(result['recommendLevel'])
            hit = [result['highFrequencyEvent'], result['occupy'], result['newEvent'], result['newEvent'], result['noOccurDays'],
                    result['periodicEvent'], result['periodicTime'], result['periodFeature'], result['periodInform']]
            hits.append(hit)

            if e-s > max_time:
                max_time = e-s
            print('max time: ', max_time)
            for watcher in watchers:
                watcher.action(result)
        e0 = time.time()
        print('data length', len(data))    # 112696
        print('total time：', e0-s0)    # 624s
        print('>0.09: ', long_length)
         # 900+  300+  568.7  884.9   352   337.7   357.5   436   360   854

        data_ori['ents'] = ents
        data_ori['hits'] = hits
        # data['recommendLevels'] = recommendLevels
        data_ori.to_csv('ent_上海银行2019_01-03去重.csv', index=False)
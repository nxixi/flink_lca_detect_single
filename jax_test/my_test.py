import pandas as pd
import time
import datetime
import pickle
import os
import matplotlib.pyplot as plt
from flink_lca_detect_single.LcaDetectSingle import *


def time2stamp(tim):
    timeArray = time.strptime(str(tim), "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp * 1000


def stamp2time(stamp):  # 时间转换：1602259200 -> '2020-10-10 00:00:00'
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stamp))


def plot_res(data, file_name=None, sig=3):
    data['day'] = data['@timestamp'].apply(lambda x: x.split(' ')[0])
    data.index = pd.to_datetime(data['@timestamp'])
    for key, group in data.groupby('template_id'):
        upper_anomaly_index = group[group['is_upper_anomaly']].index.tolist()
        lower_anomaly_index = group[group['is_lower_anomaly']].index.tolist()

        plt.figure(figsize=(30, 15))
        h1, = plt.plot(group['@value'], color='#34A5DA')
        sca_plot_index = group[group['@value'] > 0].index.tolist()
        plt.scatter(sca_plot_index, group['@value'].loc[sca_plot_index], marker='o')

        # h2, = plt.plot(day_dat['mean_list'], color='lime')
        # plt.fill_between(day_dat.index, day_dat['mean_std_thresh_upper'], day_dat['mean_std_thresh_lower'],
        #                  color='paleturquoise', alpha=0.5)
        h6 = plt.scatter(upper_anomaly_index, group['@value'].loc[upper_anomaly_index], color='red', marker='o', s=60)
        h7 = plt.scatter(lower_anomaly_index, group['@value'].loc[lower_anomaly_index], color='lime', marker='o', s=60)
        plt.legend([h1, h6, h7], ['@value', 'upper_anomaly', 'lower_anomaly'], fontsize=30)
        # plt.legend([h1], ['@value', 'upper_anomaly', 'lower_anomaly'])

        plt.title('sigma=' + str(sig), size=35)
        plt.xlabel('timestamp', fontdict={'size': 35})
        plt.ylabel('value', fontdict={'size': 35})
        plt.xticks(size=20)
        plt.yticks(size=20)

        if file_name:
            path = 'plot/' + file_name
            if not os.path.exists(path):
                os.mkdir(path)
            plt.savefig(path + '/' + str(key) + '.png')
        else:
            plt.show()
        plt.close()


def process_and_detect():
    config = {'up_alg_enable': True, 'up_thresh_enable': False, 'up_sigma': 3, 'up_thresh': 1000,
              'down_alg_enable': True, 'down_thresh_enable': False, 'down_sigma': 3, 'down_thresh': 10}
    data = pd.read_csv('test_data/my_test_data.csv')
    data = data.sort_values(by='@timestamp')
    data = data.reset_index(drop=True)

    model = {}
    input_config = {"id_field": 'id', 'time_field': 'timestamp', 'value_field': 'value',
              'abnormal_type_field': 'abnormal_type', 'value_his_field': 'value_his',
              'train_res_field': 'train_res', 'detection_config_field': 'detection_config'}
    is_upper_anomaly = []
    is_lower_anomaly = []
    anomaly_type = []
    abnormal_type = []
    for index, row in data.iterrows():
        template_id = row['template_id']
        dat = data.iloc[:index+1]
        dat = dat[dat['template_id'] == template_id]
        dat.index = pd.to_datetime(dat['@timestamp'])
        value_his = list(dat['@value'].resample('5min').sum())[:-1]
        if len(value_his) > 288*2:
            train_date = pd.to_datetime(row['@timestamp']).date() - datetime.timedelta(days=1)
            with open('/Users/fangwang/apple/PycharmProject/demo-master/日志量异常检测/spark_lca_train/jax_test/out/' + str(train_date) + '.pkl', 'rb') as f:
                train_res = pickle.load(f)
            input = {'id': template_id, 'timestamp': time2stamp(row['@timestamp']), 'value': row['@value'],
                     'abnormal_type': -1, 'value_his': value_his, 'train_res': train_res[template_id], 'detection_config': config}
            if template_id not in model:
                lca_detect_single = LcaDetectSingle()
                lca_detect_single.configure(input_config)
                model[template_id] = lca_detect_single
            else:
                lca_detect_single = model[template_id]
            result = lca_detect_single.transform(input)
            is_upper_anomaly.append(result['result']['is_upper_anomaly'])
            is_lower_anomaly.append(result['result']['is_lower_anomaly'])
            anomaly_type.append(result['result']['anomaly_type'])
            abnormal_type.append(result['abnormal_type'])
        else:
            is_upper_anomaly.append(False)
            is_lower_anomaly.append(False)
            anomaly_type.append('')
            abnormal_type.append(-1)
    data['is_upper_anomaly'] = is_upper_anomaly
    data['is_lower_anomaly'] = is_lower_anomaly
    data['anomaly_type'] = anomaly_type
    data['abnormal_type'] = abnormal_type
    data.to_csv('out/my_test_data_result.csv', index=False)

    return data


if __name__ == '__main__':

    data = process_and_detect()
    # data = pd.read_csv('out/my_test_data_result.csv')
    plot_res(data, file_name='my_test_data', sig=3)
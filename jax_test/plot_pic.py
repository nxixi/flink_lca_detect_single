#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
__author__ = 'nxx'

import numpy as np
import seaborn as sns
# from pyecharts import HeatMap

from pyecharts.charts import HeatMap
from pyecharts.charts import Scatter
from pyecharts import options as opts
import matplotlib.pyplot as plt
from jax_test.my_test import *


def plot_echarts_heatmap(id, bucket, file_name, max_, x_axis, y_axis):
    c = (
        HeatMap(opts.InitOpts(width="1500px", height="700px"))
            .add_xaxis(x_axis)
            .add_yaxis(str(id) + "热力图", y_axis, value=bucket, label_opts=[[i[0], i[1], 10] for i in bucket])
            .set_global_opts(
            title_opts=opts.TitleOpts(title="HeatMap-基本示例", subtitle="", title_textstyle_opts=opts.TextStyleOpts(font_size=30)),
            xaxis_opts=opts.AxisOpts(name="日期", axislabel_opts=opts.LabelOpts(font_size=16), name_textstyle_opts=opts.TextStyleOpts(font_size=16)),  # x轴数据项单位字体调整
            yaxis_opts=opts.AxisOpts(name="小时", axislabel_opts=opts.LabelOpts(font_size=16), name_textstyle_opts=opts.TextStyleOpts(font_size=16)),  # y轴数据项字体调整
            visualmap_opts=opts.VisualMapOpts(max_=int(max_)),

        )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="inside", font_size=11),
                             areastyle_opts=opts.AreaStyleOpts())
            .render('test_data/my_test_data_plt/' + file_name + '/' + str(id) + '.html')
    )

    # heatmap = HeatMap(width=1600, height=600)
    # heatmap.add(
    #     str(id) + "热力图",
    #     x_axis,
    #     y_axis,
    #     np.flipud(bucket),
    #     is_visualmap=True,
    #     visual_text_color="#000",
    #     visual_orient="horizontal",
    # )
    # heatmap.set_global_opts(visualmap_opts=opts.VisualMapOpts(max_=47))
    # heatmap.render('test_data/my_test_data_plt/' + file_name + '/' + str(id) + '.html')

    a = 1

def plot(id, bucket, file_name, is_save):
    # plt.figure(figsize=(80, 25))
    # sns.set(font_scale=3,font='STSong')
    ax = sns.heatmap(bucket, annot=True, center=0)
    ax.set_title(id)
    ax.set_xlabel('day')
    # ax.set_xticklabels([])  # 设置x轴图例为空值
    ax.set_ylabel('hour')
    if is_save:
        plt.savefig('test_data/my_test_data_plt/' + file_name + '/' + str(id) + '.png')
    else:
        plt.show()
    plt.close()

def run(data_name):
    data = read_data(data_name, selected=True)
    data = data.sort_values(by='timestamp')

    data['date'] = data['timestamp'].apply(lambda x: x.split(' ')[0])
    data['time'] = data['timestamp'].apply(lambda x: pd.to_datetime(x).time())
    # 得到开始到结束的天
    data['date'] = data['timestamp'].apply(lambda x: x.split(' ')[0])
    start_ = pd.to_datetime(data['date'].min()).date()
    end_ = pd.to_datetime(data['date'].max()).date()
    all_dates = get_days(start_, end_)


    # shanghai
    data['template_id'] = data['template_id'].apply(lambda x: str(x))
    data['ip'] = data['ip'].apply(lambda x: str(x))
    data['grouping'] = data['ip'] + '|' + data['template_id']

    # pufa
    # nanjing
    # jiaohang
    # xmgj
    # union
    # dfzq


    for key, group in data.groupby('grouping'):
        s = time.time()
        group['timestamp'] = pd.to_datetime(group['timestamp'])
        group['hour'] = group['timestamp'].apply(lambda x: x.hour)
        gro = group[['timestamp']]
        gro['count'] = 1
        gro.index = gro['timestamp']
        gro_af = gro.resample('1h').count()[['count']]
        gro_af['timestamp'] = gro_af.index
        gro_af = gro_af.reset_index(drop=True)
        gro_af['day'] = gro_af['timestamp'].apply(lambda x: x.date())
        gro_af['hour'] = gro_af['timestamp'].apply(lambda x: x.hour)
        days = set(gro_af['day'])
        dic1 = {i: gro_af[gro_af['day']==i] for i in days}
        dic2 = {i: dict(zip(dic1[i]['hour'], dic1[i]['count'])) for i in dic1}
        dic2 = {i: {j: 0 if j not in dic2[i] else dic2[i][j] for j in range(24)} for i in dic2}
        sort_dic2 = sorted(dic2.items(), key=lambda d: d[0], reverse=False)
        dic3 = {i[0]: i[1] for i in sort_dic2}
        bucket = np.array([list(dic3[i].values()) if i in dic3 else [0] * 24 for i in all_dates])
        e = time.time()
        print(e-s)

        # group['date'] = group['timestamp'].apply(lambda x: x.split(' ')[0])
        # group['time'] = group['timestamp'].apply(lambda x: str(pd.to_datetime(x).time()))

        plot(str(key), bucket.T, data_name, is_save=True)

        a = 1

def run1(data_name):
    data = read_data(data_name, selected=True)
    data = data.sort_values(by='timestamp')
    data['date'] = data['timestamp'].apply(lambda x: x.split(' ')[0])
    data['time'] = data['timestamp'].apply(lambda x: pd.to_datetime(x).time())
    # 得到开始到结束的天
    data['date'] = data['timestamp'].apply(lambda x: x.split(' ')[0])
    start_ = pd.to_datetime(data['date'].min()).date()
    end_ = pd.to_datetime(data['date'].max()).date()
    all_dates = get_days(start_, end_)

    # shanghai
    data['template_id'] = data['template_id'].apply(lambda x: str(x))
    data['ip'] = data['ip'].apply(lambda x: str(x))
    data['grouping'] = data['ip'] + '|' + data['template_id']

    data['count'] = 1
    group_count = data.groupby('grouping')['count'].count()


    import plotly.express as px
    fig = px.histogram(group_count)
    fig.show()

    a = 1



if __name__ == '__main__':

    data = pd.read_csv('../jax_test/test_data/my_test_data_sharp_increase1.csv')

    data['timestamp'] = data['timestamp'].apply(lambda x: str(pd.to_datetime(x)))

    data = data.sort_values(by='timestamp')

    # 得到开始到结束的天
    data['date'] = data['timestamp'].apply(lambda x: x.split(' ')[0])
    start_ = pd.to_datetime(data['date'].min()).date()
    end_ = pd.to_datetime(data['date'].max()).date()
    all_dates = get_days(start_, end_)

    for key, group in data.groupby('template_id'):

        s = time.time()
        group['timestamp'] = pd.to_datetime(group['timestamp'])
        group['hour'] = group['timestamp'].apply(lambda x: x.hour)
        gro = group[['timestamp']]
        gro['count'] = 1
        gro.index = gro['timestamp']
        gro_af = gro.resample('1h').count()[['count']]
        gro_af['timestamp'] = gro_af.index
        gro_af = gro_af.reset_index(drop=True)
        gro_af['day'] = gro_af['timestamp'].apply(lambda x: x.date())
        gro_af['hour'] = gro_af['timestamp'].apply(lambda x: x.hour)
        days = set(gro_af['day'])
        dic1 = {i: gro_af[gro_af['day']==i] for i in days}
        dic2 = {i: dict(zip(dic1[i]['hour'], dic1[i]['count'])) for i in dic1}
        dic2 = {i: {j: 0 if j not in dic2[i] else dic2[i][j] for j in range(24)} for i in dic2}
        sort_dic2 = sorted(dic2.items(), key=lambda d: d[0], reverse=False)
        dic3 = {i[0]: i[1] for i in sort_dic2}
        bucket = np.array([list(dic3[i].values()) if i in dic3 else [0] * 24 for i in all_dates])
        e = time.time()
        print(e-s)

        plot(str(key), bucket.T, 'my_test_sharp_increase1', is_save=True)

    datas = ['shanghai', 'pufa', 'nanjing', 'jiaohang', 'xmgj', 'union', 'dfzq']
    run1('shanghai')
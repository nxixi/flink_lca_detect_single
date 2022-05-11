#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
__author__ = 'nxx'

import pandas as pd
import matplotlib.pyplot as plt
from jax_test.my_test import *
from jax_test.plot_pic import *


## 得到开始到结束的天
def get_all_days(data):
	data['date'] = data['time'].apply(lambda x: x.split(' ')[0])
	start_ = pd.to_datetime(data['date'].min()).date()
	end_ = pd.to_datetime(data['date'].max()).date()
	all_dates = get_days(start_, end_)
	return all_dates, start_, end_

##
def get_bucket(group, all_dates):
	group['timestamp'] = pd.to_datetime(group['time'])
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
	dic1 = {i: gro_af[gro_af['day'] == i] for i in days}
	dic2 = {i: dict(zip(dic1[i]['hour'], dic1[i]['count'])) for i in dic1}
	dic2 = {i: {j: 0 if j not in dic2[i] else dic2[i][j] for j in range(24)} for i in dic2}
	sort_dic2 = sorted(dic2.items(), key=lambda d: d[0], reverse=False)
	dic3 = {i[0]: i[1] for i in sort_dic2}
	bucket = np.array([list(dic3[i].values()) if i in dic3 else [0] * 24 for i in all_dates])
	return bucket.T

def bucket_transform(bucket):
	bucket_list = bucket.tolist()
	bucket_for_plot_echarts = [[i,j,bucket_list[j][i]] for i in range(bucket.shape[1]) for j in range(bucket.shape[0])]

	return bucket_for_plot_echarts


def label_occupy_analyze(result_data):

	no_label_data = result_data[result_data['describe'] == 'nan']
	print('无标签告警数量：' + str(len(no_label_data)) + '，占比为：' + str(round(len(no_label_data)/len(result_data), 4)) + '，共' + str(len(no_label_data['group_id'].unique())) + '类告警')
	label_data = result_data[result_data['describe'] != 'nan']
	print('有标签告警数量：' + str(len(label_data)) + '，占比为：' + str(round(len(label_data)/len(result_data), 4)) + '，共' + str(len(label_data['group_id'].unique())) + '类告警')

	def sta(result_data, label, label_name):
		label_data = result_data.loc[result_data['describe'].str.contains(label)]
		class_num = len(label_data['group_id'].unique())
		print(label_name + '告警数量：' + str(len(label_data)) + '，占比为：' + str(round(len(label_data) / len(result_data), 4)) + '，共' + str(class_num) + '类告警')
		return class_num

	occu_new_class_num = sta(result_data, '新增：', '新增')
	occu_new_tem_class_num = sta(result_data, '新增模版', '新增模版')
	occu_rare_class_num = sta(result_data, '偶发：', '偶发')
	occu_new_tem_class_num = sta(result_data, '偶发模版', '偶发模版')
	occu_sharp_increase_class_num = sta(result_data, '激增', '激增')
	occu_high_frequency_class_num = sta(result_data, '高频', '高频')
	occu_period_class_num = sta(result_data, '周期', '周期')
	print('=========================================================================')

	labels = {}
	for key, group in result_data.groupby('label'):
		if key != 'nan':
			classes = list(group['group_id'].unique())
			print(key + '告警数量：' + str(len(group)) + '，占比为：' + str(round(len(group) / len(result_data), 4)) + '，共' + str(
				len(classes)) + '类告警')
			labels[key] = classes
	return labels


# def label_distribute_analyze(result_data):
# 	labeled_data = result_data[result_data['time'] >= '2019-01-31 00:00:00']
#
# 	import plotly.express as px
# 	fig = px.histogram(data_hist, x="date")
#
# 	a = 1


def label_plot():
	all_dates, start_, end_ = get_all_days(result_data)
	x_axis = [str(i) for i in all_dates]
	y_axis = [i for i in range(24)]

	sub_ = 30
	plt.figure(figsize=(100, 100))
	plt.subplots_adjust(left=0.03, bottom=0.01, right=1, top=1, wspace=0.07, hspace=0.05)
	# plt.tight_layout()
	i = 1
	for key, group in result_data.groupby('group_id'):
		# if not group.loc[group['describe'].str.contains('高频')].empty:
		if not group[group['label'] == '新增'].empty and i <= 60:
		# if (not group.loc[group['describe'].str.contains('激增')].empty) or (not group.loc[group['describe'].str.contains('高频')].empty):
			# 统计每小时发生次数
			gro = group[['time']]
			gro['count'] = 1
			gro = gro.append(pd.DataFrame({'time': [str(start_) + ' 00:00:00', str(end_) + ' 23:59:59'], 'count': [0, 0]}))
			gro = gro.sort_values('time')
			gro.index = pd.to_datetime(gro['time'])
			gro1 = gro.resample('1h').sum()
			gro1[str(key)] = gro1['count']

			plt.subplot(sub_, 2, i)
			plt.plot(gro1[str(key)])
			plt.legend()
			i += 1
			plt.xticks(())

			# # 画热力图
			# bucket = get_bucket(group, all_dates)
			# max_ = bucket.max()
			# bucket_for_plot_echarts = bucket_transform(bucket)
			# plot_echarts_heatmap(str(key), bucket_for_plot_echarts, data_name, max_, x_axis, y_axis)
			# # plot(str(key), bucket.T, data_name, is_save=True)
			# a = 1

	plt.show()

	# for key, group in high_frequency_data.groupby('group_id'):
	# 	x = pd.to_datetime(result_data[result_data['group_id'] == key]['time'])
	# 	plt.scatter(x, [1]*len(x))
	# 	plt.show()
	# 	a = 1
	# a = 1

from matplotlib.colors import ListedColormap

def scatter_(x, y, values, classes):
	colo = ['b', 'c', 'y', 'm', 'r', 'orange', 'deeppink', 'peru', 'grey', 'orchid', 'salmon', 'gold', 'cyan', 'violet']

	colours = ListedColormap(colo)
	scatter = plt.scatter(x, y, c=values, cmap=colours)
	plt.legend(handles=scatter.legend_elements()[0], labels=classes)

	plt.show()

	a = 1



def plot_scatter(result_data, labels, lab=''):
	import plotly.express as px

	if not lab:
		for i in labels:
			group_ids = labels[i]
			for j in group_ids:
				dat = result_data[result_data['group_id']==j]

				scatter_(x=dat['time'], y=dat['label_value'], values=dat['label_value'], classes=set(dat['label']))

				plt.scatter(dat['time'], dat['label_value'], c=set(dat['label_value']))
				plt.legend()
				plt.show()
				a = 1
	else:
		group_ids = labels[lab]
		for j in group_ids:
			dat = result_data[result_data['group_id'] == j]
			plt.scatter()



def res_process(data_name):
	result_data = pd.read_csv('../jax_test/test_data/result/' + data_name + '_ent_new30_high30_increase30_1.csv')

	result_data['describe'] = result_data['describe'].apply(lambda x: str(x))
	result_data['time'] = result_data['timestamp'].apply(lambda x: stamp2time(x//1000))
	result_data['label'] = result_data['describe'].apply(lambda x: '、'.join(sorted(set([i.split('：')[0] for i in x.split(';\n') if i != '']))))   # 仅取标签
	result_data['label'] = result_data['label'].apply(lambda x: str(x))
	result_data['date'] = result_data['time'].apply(lambda x: x.split(' ')[0])

	labs = list(result_data['label'].unique())
	labels_map = dict(zip(labs, range(1, len(labs) + 1)))
	result_data['label_value'] = result_data['label'].apply(lambda x: labels_map[x])

	return result_data


if __name__ == '__main__':
	data_name = '上海银行2019_01-03去重'
	result_data = res_process(data_name)
	labels = label_occupy_analyze(result_data)
	# label_distribute_analyze(result_data)
	# plot_scatter(result_data, labels, lab='偶发、偶发模版、新增、新增模版')

	all_dates, start_, end_ = get_all_days(result_data)
	x_axis = [str(i) for i in all_dates]
	y_axis = [i for i in range(24)]

	# for lab in labels:
	# 	for i in labels[lab]:
	# 		group = result_data[result_data['group_id'] == i]
	#
	# 		# # 画热力图
	# # 			# bucket = get_bucket(group, all_dates)
	# # 			# max_ = bucket.max()
	# # 			# bucket_for_plot_echarts = bucket_transform(bucket)
	# # 			# plot_echarts_heatmap(str(i), bucket_for_plot_echarts, data_name, max_, x_axis, y_axis)
	# # 			# plot(str(i), bucket.T, data_name, is_save=False)
	# 		a = 1
	# 	a = 1

	ids = ['10.239.2.24|4', '10.239.2.24|5', '10.232.84.120|6', '181.5.2.34|1']
	for i in ids:
		key = i
		group = result_data[result_data['group_id'] == key]

		# key = i.split('|')[1]
		# group = result_data[result_data['template_id'] == int(key)]

		# 画热力图
		bucket = get_bucket(group, all_dates)
		max_ = bucket.max()
		bucket_for_plot_echarts = bucket_transform(bucket)
		plot_echarts_heatmap(str(key), bucket_for_plot_echarts, data_name, max_, x_axis, y_axis)
		# plot(str(i), bucket.T, data_name, is_save=False)

		a = 1


	counts = {}
	for key, group in result_data.groupby('group_id'):
		if len(group) in counts:
			counts[len(group)].append(key)
		else:
			counts[len(group)] = [key]
	counts1 = {i: len(counts[i]) for i in counts}

	df1 = result_data[(result_data['label'].str.contains('新增'))]
	df2 = df1[~df1['label'].str.contains('新增模版')]
	print('新增：' + str(len(df1)) + '条，新增但不是新增模版：' + str(len(df2)) + '条，后者占前者的比例为' + str(round(len(df2)/len(df1), 2)))

	df3 = result_data[(result_data['label'].str.contains('偶发'))]
	df4 = df3[~df3['label'].str.contains('偶发模版')]
	print('偶发：' + str(len(df3)) + '条，偶发但不是偶发模版：' + str(len(df4)) + '条，后者占前者的比例为' + str(round(len(df4)/len(df3), 2)))

	df5 = result_data[(result_data['label'].str.contains('偶发模版'))]
	df6 = df5[~df5['label'].str.contains('偶发')]
	print('偶发模版：' + str(len(df5)) + '条，偶发模版但不是偶发：' + str(len(df6)) + '条，后者占前者的比例为' + str(round(len(df6)/len(df5), 2)))

	i = 0
	for key, group in result_data.groupby('group_id'):
		if (group[group['label'].str.contains('激增')].empty and group[group['label'].str.contains('高频')].empty and group[group['label'].str.contains('周期')].empty):
		# if (not group[group['label'].str.contains('激增')].empty):

		# if len(group) < 30 and ((not group[group['label'].str.contains('激增')].empty) or (not group[group['label'].str.contains('高频')].empty) or (not group[group['label'].str.contains('周期')].empty)):
			i += 1

		a = 1

	a = 1

'新增模版、偶发模版、偶发、新增'

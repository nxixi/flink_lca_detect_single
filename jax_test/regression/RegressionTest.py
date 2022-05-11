# -*- coding: utf-8 -*-


import unittest
import sys
import os
import pandas as pd
import numpy as np
from regression import Regression

# add path of jax-algorithm to the sys.path
SCRIPT_PATH=os.path.split(os.path.realpath(__file__))[0]
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_PATH)))


class Ripple_test(unittest.TestCase):

    def tearDown(self):
        '''
        ###每个测试用例执行之后做操作
        :return:
        '''
        pass

    def setUp(self):
        '''
        ###每个测试用例执行之前做操作
        :return:
        '''
        pass

    @classmethod
    def tearDownClass(self):
        '''
        ###必须使用 @ classmethod装饰器, 所有test运行完后运行一次
        :return:
        '''
        pass

    @classmethod
    def setUpClass(self):
        '''
        ###必须使用@classmethod 装饰器,所有test运行前运行一次
        :return:
        '''
        pass

    def get_test_file(self,filename):
        return os.path.join(os.path.split(os.path.realpath(__file__))[0],filename)

    def test_regression1(self):

        data=pd.DataFrame({"X":[1,2,3,4,5,6,7,8,9,10],
                           "Y":[3,5,7,9,11,13,15,17,19,21]
                           })
        regressionAlg=Regression.Regression()
        params=dict(X='X',Y='Y',poly_n=1,assign_x=100,assign_y=2001)
        regressionAlg.configure(params)
        result,model=regressionAlg.transform(data)
        self.assertEqual(round(result['x_predict'].iloc[0],2),1000)
        self.assertEqual(round(result['y_predict'].iloc[0],2),201)
        # import matplotlib.pyplot as plt
        # result = result.sort_values(by=['X'])
        # plt.figure(figsize=(12, 8))
        # plt.scatter(result['X'], result['Y'])
        # plt.plot(result['X'], result['fit'], 'r', linestyle='--')
        # plt.show()


    def test_regression2(self):

        data=pd.DataFrame({"X":[1,2,3,4,5,6,7,8,9,10],
                           "Y":[2,5,10,17,26,37,50,65,82,101]
                           })
        regressionAlg=Regression.Regression()
        params=dict(X='X',Y='Y',poly_n=2,assign_x=100,assign_y=401)
        regressionAlg.configure(params)
        result,model=regressionAlg.transform(data)
        self.assertEqual(round(result['x_predict'].iloc[0],2),20)
        self.assertEqual(round(result['y_predict'].iloc[0],2),10001)
        # import matplotlib.pyplot as plt
        # result = result.sort_values(by=['X'])
        # plt.figure(figsize=(12, 8))
        # plt.scatter(result['X'], result['Y'])
        # plt.plot(result['X'], result['fit'], 'r', linestyle='--')
        # plt.show()


if __name__ == '__main__':
    unittest.main()

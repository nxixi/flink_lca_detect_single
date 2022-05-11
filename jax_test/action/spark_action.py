from pyspark import SparkContext, SQLContext

from jax_python.aad import AbstractAbnormalDetect
from jax_python.shim import Shim
from regression.Regression import Regression


def run(aad:AbstractAbnormalDetect, conf_dict, group_by_fields=None):
    if group_by_fields is None:
        group_by_fields = []
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
        'pivot_test_slim.csv')

    df = Shim.run(sqlContext,df,group_by_fields,aad,conf_dict)
    return df


if __name__ == '__main__':

    job = Regression()
    conf_dict = {
        "X":"AMP",
        "Y":"value"
    }
    df = run(job,conf_dict)
    df.show(100)
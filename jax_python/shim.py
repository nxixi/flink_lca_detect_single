# this module is a util for python communicate with spark
# at least, PYTHONPATH must include the following path before import and run:
# $SPARK_HOME/python
# $SPARK_HOME/python/lib/py4j-0.10.7-src.zip

import os
import io
import sys
import traceback
import json
import time
import pandas as pd
import numpy as np
from pyspark.sql.types import StructField, StructType, IntegerType, FloatType, TimestampType, StringType, BooleanType, DoubleType, LongType
from pyspark.context import SparkConf, SparkContext
from pyspark.sql import SQLContext, HiveContext, SparkSession, DataFrame, SQLContext
from pyspark.sql.functions import collect_list, struct
from pyspark import AccumulatorParam, Accumulator
from py4j.java_gateway import JavaGateway, java_import, GatewayClient, GatewayParameters, JavaObject
from py4j.java_collections import JavaMap, JavaSet, JavaArray, JavaList, JavaIterator
from contextlib import redirect_stdout, redirect_stderr

# PYSPARK_ENV extract and store important environment variable
# jvm must start the java gateway, and fill some environment variable before
# start the python process which likely comminucate with jvm as well as spark object
#
# PYTHONPATH: the path that python used to finding package and module
# PY4J_SECRET: the pj4y's java gateway server secret
# PY4J_PORT: the pj4y's java gateway port
# sys.version: the python version

PYSPARK_ENV = {
    'PYTHONPATH': os.getenv('PYTHONPATH'),
    'PY4J_SECRET': os.getenv('PY4J_SECRET'),
    'PY4J_PORT': os.getenv('PY4J_PORT'),
    'sys.version': sys.version,
    '_DISABLE_LOG_ACCUMULATOR': os.getenv('_DISABLE_LOG_ACCUMULATOR')
}


# print the PYSPARK_ENV is useful for debug some issue about environment
def printEnv():
    print("python execution information: ", PYSPARK_ENV)


def type_literal_to_sf(tl):
    if tl == 'int':
        return IntegerType()
    if tl == 'long':
        return LongType()
    elif tl == 'float':
        return FloatType()
    elif tl == 'double':
        return DoubleType()
    elif tl == 'str':
        return StringType()
    elif tl == 'boolean':
        return BooleanType()


def get_field_schema_list(fields):
    fieldSchemaList = []
    for field in fields:
        fieldSchemaList.append(StructField(field[0], type_literal_to_sf(field[1]), nullable=True))
    return fieldSchemaList


class StringCollectionAccumulatorParam(AccumulatorParam):

    def zero(self, value):
        return []

    def addInPlace(self, value1, value2):
        # merge two list
        if isinstance(value2, str):
            value1.append(value2)
            return value1
        elif isinstance(value2, list):
            return value1 + value2
        else:
            return value1

def javaToPython(java_object):
    """
    input is a JavaObject which generated by py4j
    :param java_object:
    :return: a python variable
    """
    if java_object is None:
        return None
    elif isinstance(java_object, JavaMap):
        dict = {}
        for key in java_object:
            dict[key] = javaToPython(java_object[key])
        return dict
    elif isinstance(java_object, JavaList) or isinstance(java_object, JavaSet) or isinstance(java_object, JavaIterator):
        list = []
        for item in java_object:
            list.append(javaToPython(item))
        return list
    elif isinstance(java_object, JavaArray):
        list = []
        for i in range(len(java_object)):
            list.append(javaToPython(java_object[i]))
        return list
    elif isinstance(java_object, JavaObject):
        # TODO JavaObject ?????????????????????????????????????????????POJO???
        return None
    else:
        return java_object

class Shim:
    """
    entry_point
    spark_conf
    spark_context
    spark_session
    sql_context
    """

    def __init__(self):
        """
        init the connection to the JavaGateway by PY4J_PORT and PY4J_SECRET
        return: the EntryPoint stub which defined in jvm, access jvm from the EntryPoint
        the EntryPoint must be sub class of com.eoi.jax.job.spark.process.PythonJobEndpoint
        """
        port = int(PYSPARK_ENV['PY4J_PORT'])
        params = GatewayParameters(port=port, auto_convert=True, auth_token=PYSPARK_ENV['PY4J_SECRET'])
        gateway = JavaGateway(gateway_parameters=params)
        ep = gateway.entry_point
        imports = ep.getPy4JImports()
        for i in imports:
            java_import(gateway.jvm, i)
        self.entry_point = ep
        self.spark_conf, self.spark_context, self.spark_session, self.sql_context = Shim._context(gateway, ep)

    def log_accumulator(self):
        """
        create a string collection accumulator, used for accumulate python log from lambda(run on spark worker)
        :return: a string collection accumulator
        """
        return self.spark_context.accumulator([], StringCollectionAccumulatorParam())

    def wait_for_stop(self):
        """
        block and wait until the spark context stopped
        :return:
        """
        while True:
            time.sleep(1)
            # https://stackoverflow.com/questions/36044450/how-to-check-that-the-sparkcontext-has-been-stopped/36044685
            if self.spark_context._jsc.sc().isStopped():
                return

    def py_data_frame(self, jdf):
        """
        convert jvm DataFrame to python wrapped DataFrame
        :param jdf: jvm DataFrame(scala)
        :return: pyspark.sql.DataFrame
        """
        return DataFrame(jdf, self.sql_context)

    def py_list(self, jvm_list):
        """
        convert jvm List to python list
        :param jvm_list: jvm List
        :return: python llist
        """
        r = []
        if jvm_list is not None:
            for i in jvm_list:
                r.append(i)
        return r

    def get_input0(self):
        """
        get the first input jvm DataFrame from entry_point
        :return: the first input jvm DataFrame
        """
        return self.entry_point.input()

    def set_output0(self, data_frame):
        """
        set the output as Python DataFrame
        auto convert to jvm DataFrame
        :param data_frame: Python DataFrame
        """
        self.entry_point.output().add(data_frame._jdf)

    @staticmethod
    def run(sql_context, df, group_by_fields, detector, conf_dict, log_acc=None):
        # http://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html

        # no arrow enabled case
        group_by_fields_schema = []
        non_group_by_fields_schema = []

        for sf in df.schema:
            if sf.name in group_by_fields:
                group_by_fields_schema.append(sf)
            else:
                non_group_by_fields_schema.append(sf)
        group_by_fields = [s.name for s in group_by_fields_schema]

        # pass in columns except groupByFields
        pass_in_cols = df.columns
        for gf in group_by_fields:
            pass_in_cols.remove(gf)

        # prepare schema
        if detector.mode_append():
            # mode_append????????????????????????
            schema = StructType(non_group_by_fields_schema)
            for sf in get_field_schema_list(detector.fields()):
                if sf not in schema:
                    schema.add(sf)
        else:
            # mode_append=False??????????????????????????????????????????
            schema = StructType(get_field_schema_list(detector.fields()))
            # ????????????groupBy??????????????????
        for sf in group_by_fields_schema:
            schema.add(sf)
        print(schema)  # [DRIVER]

        df = df.groupBy(group_by_fields).agg(collect_list(struct(pass_in_cols)).alias('data'))
        df = df.rdd.map(lambda r: Shim._transform_row(r, pass_in_cols)).map(
            lambda d: Shim._wrap_func(d, detector, group_by_fields, PYSPARK_ENV, conf_dict, log_acc))
        df = df.flatMap(lambda r: Shim._expand_row(r, group_by_fields, schema))
        df = sql_context.createDataFrame(df, schema)
        return df

    @staticmethod
    def _transform_row(row, pass_in_cols):
        """
        Transform row(DataRow) into a dict, and transform the row['data'] from list into pandas.DataFrame
        :param row: pyspark.sql.Row
        :return:
        """
        list_of_data = row['data']
        # each d in list_of_data should be a struct
        # NOTE: pd.DataFrame will auto reorder the columns by dictionary order
        # so we have to avoid this by explictily pass pass_in_cols as columns
        data = pd.DataFrame([d.asDict() for d in list_of_data], columns=pass_in_cols)
        # print(data.columns) # [EXECUTOR]
        dict = row.asDict()
        dict['data'] = data
        return dict

    @staticmethod
    def _expand_row(dict, group_by_fields, schema):

        """
        expand the dict['data'](which is the result pandas.DataFrame), merge groupByFields at the same time
        finally return list of tuple
        :param dict:
        :param group_by_fields:
        :param schema, spark schema
        :return:
        """
        pdf = dict['data']
        for gf in group_by_fields:
            pdf[gf] = dict[gf]
        tuple_list = np.array(pdf).tolist()
        timestamp_column_index = []
        for i in range(len(pdf.columns)):
            col_name = pdf.columns[i]
            if col_name in schema.names and isinstance(schema[col_name].dataType, TimestampType):
                timestamp_column_index.append(i)

        for item in tuple_list:
            for i in timestamp_column_index:
                item[i] = item[i].to_pydatetime()

        return tuple_list

    @staticmethod
    def _wrap_func(dict, detector, group_by_fields, env, conf_dict, log_acc=None):
        # group_info = str(",".join([str(dict[gf]) for gf in group_by_fields]))
        group_bys = {gf: dict[gf] for gf in group_by_fields}
        print("group_bys: ", group_bys)
        panda_df = dict['data']

        error_string = ''
        fout = None
        ferr = None
        if not env.__contains__('_DISABLE_LOG_ACCUMULATOR'):
            # normal case: redirect stdout and stderr to string if _DISABLE_LOG_ACCUMULATOR not set
            # in debug situation, spark run in local mode, we usually need to see `print` result directly
            fout = io.StringIO()
            ferr = io.StringIO()
            redirect_stdout(fout)
            redirect_stderr(ferr)
        try:
            start = time.time()
            result_df, model = detector.transform(panda_df)
            end = time.time()
            print("group ", group_bys , " transform cost " + str(end - start))
            if model is not None:
                backend = init_backend(conf_dict)
                backend.set_state(group_bys, model)
        except:
            e_type, e_value, e_traceback = sys.exc_info()
            error_string = str(''.join(traceback.format_exception(e_type, e_value, e_traceback)))
            error_string = "failed group: " , group_bys , ". " + error_string
            if log_acc:
                log_acc.add(error_string)
            # append mode should append additional fields
            if detector.mode_append():
                result_df = dict['data']
                for field in detector.fields():
                    result_df[field[0]] = None
            # otherwise should build entire dataframe based on fields
            else:
                result_df = pd.DataFrame(columns=[f[0] for f in detector.fields()])
        finally:
            if log_acc and fout and ferr:
                [log_acc.add(x) for x in fout.getvalue().splitlines()]
                [log_acc.add(x) for x in ferr.getvalue().splitlines()]
        if fout and ferr:
            fout.close()
            ferr.close()
        dict['data'] = result_df
        if error_string != '':
            print(error_string)
        return dict

    @staticmethod
    def _context(gateway, entry_point):
        """
        get context from gateway and entry_point(com.eoi.jax.job.spark.process.PythonJobEndpoint)
        return python wrap of spark context stuff
        :param gateway: the JavaGateway
        :param entry_point: the EntryPoint get by init()
        :return: SparkConf, SparkContext, SparkSession, SQLContext
        """
        builder_context = entry_point.sparkBuilderContext()
        java_spark_context = builder_context.getJavaSparkContext()
        spark_session = builder_context.getSparkSession()
        spark_conf = builder_context.getSparkConf()
        sql_context = builder_context.getSqlContext()
        # sparkConf.set("spark.sql.execution.arrow.enabled", "true")
        py_spark_conf = SparkConf(_jconf=spark_conf)
        py_spark_context = SparkContext(gateway=gateway, jsc=java_spark_context, conf=py_spark_conf)
        py_spark_session = SparkSession(py_spark_context, spark_session)
        py_sql_context = SQLContext(py_spark_context, py_spark_session, sql_context)
        return py_spark_conf, py_spark_context, py_spark_session, py_sql_context


class RedisStateBackend(object):

    def __init__(self, config):
        from redis.sentinel import Sentinel
        import redis
        """
        :param config:
            mode: single|sentinel
            hosts: host1:port1,host2:port2
            password: auth password
            master: for sentinel mode
            keyPattern: ???????????????????????????????????????redis??????????????????keyPattern??????????????????key?????????: 182h9aghag_${item_guid}?????????
                    ???????????????????????????item_guid???????????????????????????key????????????????????????????????????${}???????????????????????????????????????groupByFields???
                    ???????????????keyPattern?????????????????????????????????????????????????????????????????????key 
        """
        hosts = []
        for i, host in enumerate(config.get('hosts', []).split(',')):
            splitted = host.split(":")
            hosts.append((splitted[0], int(splitted[1])))
        if config.get('mode') == "sentinel":
            sentinel = Sentinel(hosts, password=config.get('password'))
            self.connection = sentinel.master_for(config.get('master'))
        else:
            self.connection = redis.Redis(host=hosts[0][0], port=hosts[0][1], password=config['password'])
        self.key_pattern = config.get('keyPattern', '')

    def get_key(self, group_bys):
        key = self.key_pattern
        for (k,v) in  group_bys.items():
            key = key.replace('${' + k + '}', v)
        return key

    # group_bys
    # {'item_guid': 'aa', 'instance_name': 'metric'} or {}
    def set_state(self, group_bys, binary):
        key = self.get_key(group_bys)
        self.connection.set(key, binary)

    def close(self):
        self.connection.close()

class RedisSensorStateBackend(RedisStateBackend):

    def __init__(self, config):
        super(RedisSensorStateBackend, self).__init__(config)

    def set_state(self, group_bys, binary):
        # TODO: compress
        # import base64
        # base64_str = str(base64.b64encode(binary))
        key = self.get_key(group_bys)
        self.connection.hmset(key, {'model_data': binary, 'update_time': int(time.time()) * 1000})

class NoOpStateBackend(object):
    def set_state(self, group_bys, binary):
        pass

    def close(self):
        pass


def init_backend(custom_config):
    backend = custom_config.get('stateBackend', 'noop')
    advance_config = custom_config.get('stateBackendAdvance', {})
    if backend == 'redis':
        print("init RedisStateBackend")
        return RedisStateBackend(advance_config)
    elif backend == 'redis-sensor':
        print("init RedisSensorStateBackend")
        return RedisSensorStateBackend(advance_config)
    else:
        print("init NoOpStateBackend")
        return NoOpStateBackend()
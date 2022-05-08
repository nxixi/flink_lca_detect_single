# -*- coding: utf-8 -*-
# need to add https://gitlab.eoitek.net/EOI/jax-algorithm
# to PYTHONPATH
import os
import importlib

print("*******************************************************************")
print("environment variable $PYTHONPATH: ", os.getenv('PYTHONPATH'))
print("environment variable $PYSPARK_PYTHON: ", os.getenv('PYSPARK_PYTHON'))
print("*******************************************************************")

import sys
import json
import traceback
from jax_python.shim import Shim, printEnv, javaToPython

def init_alg(config):
    module_name = config.getModuleName()
    alg_name = config.getAlgName()
    param_jvm_dict = config.getParamMap()
    print("import", module_name, "class", alg_name)
    print(param_jvm_dict)
    conf_dict = javaToPython(param_jvm_dict)
    import_module = importlib.import_module(module_name)
    import_alg = getattr(import_module, alg_name)()
    import_alg.configure(conf_dict)
    return import_alg, conf_dict

if __name__ == '__main__':
    printEnv()
    shim = Shim()
    try:
        config = shim.entry_point.jobConfig()
        input = shim.get_input0()
        df = shim.py_data_frame(input)
        alg, conf_dict = init_alg(config)
        log_acc = shim.log_accumulator()
        group_by_fields = shim.py_list(config.getGroupByFields())
        sparkDf = Shim.run(shim.sql_context, df, group_by_fields, alg, conf_dict, log_acc)
        shim.set_output0(sparkDf)
        shim.entry_point.sync().release()
        shim.wait_for_stop()
        for line in log_acc.value:
            # TODO: print非ascii字符的问题
            print(line)
    except:
        e_type, e_value, e_traceback = sys.exc_info()
        error_string = str(''.join(traceback.format_exception(e_type, e_value, e_traceback)))
        print(json.dumps({'error': error_string}))
        shim.entry_point.sync().release()
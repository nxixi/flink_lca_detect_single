import os
import pickle
from jax_python.shim import javaToPython
from py4j.java_gateway import JavaGateway, GatewayParameters
from io import BytesIO

port = int(os.getenv('PY4J_PORT'))
params = GatewayParameters(port=port, auto_convert=True, auth_token=os.getenv('PY4J_SECRET'))
gateway = JavaGateway(gateway_parameters=params)
ep = gateway.entry_point.get_map()
f = BytesIO()
pythonObject = javaToPython(ep)
print(pythonObject)
pickle.dump(pythonObject, f)

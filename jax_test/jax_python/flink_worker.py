import os
import sys
import socket
import json
import time
import struct
from py4j.java_gateway import JavaGateway, java_import, GatewayClient, GatewayParameters
from jax_python.shim import javaToPython

DATA_FEED = 0xf4f4
DATA_FEED_SNAPSHOT = 0xf3f3
DATA_RET = 0xf5f5
STATE_REQ = 0xdada
STATE_FEED = 0xe4e4
STATE_UPDATE = 0xe5e5

def read_int(stream):
    length = stream.read(4)
    if not length:
        raise EOFError
    return struct.unpack("!i", length)[0]

def write_int(stream, value):
    stream.write(struct.pack("!i", value))

def read_block(stream):
    l = read_int(stream)
    if l == 0:
        return None
    return stream.read(l)

def write_block(stream, binary):
    if binary is None:
        write_int(stream, 0)
    else:
        write_int(stream, len(binary))
        stream.write(binary)
    stream.flush()

def read_event(stream):
    """
    frame format
    DATA_FEED | json-len | json | key-len | key
    :param stream:
    :return: (event, key)
    """
    cmd = read_int(stream) # DATA_FEED flag
    need_snapshot = False
    if cmd == DATA_FEED_SNAPSHOT:
        need_snapshot = True
    event = json.loads(str(read_block(stream), encoding="utf-8"))
    key = str(read_block(stream), encoding="utf-8")
    return event, key, need_snapshot

def write_events(stream, events):
    """
    event count | 1st-len | json | 2nd-len | json ...
    :param stream:
    :param events: list of event
    :return:
    """
    write_int(stream, DATA_RET)
    event_len = len(events)
    write_int(stream, event_len)
    for i, val in enumerate(events):
        ser = bytes(json.dumps(val), encoding='utf-8')
        write_block(stream, ser)
    stream.flush()

def handshake(stream, secret):
    token = bytes(secret, encoding='utf-8')
    write_block(stream, token)
    read_block(stream)


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
            keyPrefix: 
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
        self.key_prefix = config.get('keyPrefix', '')

    def get_state(self, key):
        return self.connection.get(self.key_prefix + key)

    def set_state(self, key, binary):
        self.connection.set(self.key_prefix + key, binary)

class RedisSensorStateBackend(RedisStateBackend):

    def __init__(self, config):
        super(RedisSensorStateBackend, self).__init__(config)

    def get_state(self, key):
        return self.connection.hmget(self.key_prefix + key, 'model_data')[0]

    def set_state(self, key, binary):
        # TODO: compress
        # import base64
        # base64_str = str(base64.b64encode(binary))
        self.connection.hmset(self.key_prefix + key, {'model_data': binary, 'update_time': int(time.time()) * 1000})

class FlinkStateBackend(object):

    def __init__(self, stream):
        self.stream = stream

    def get_state(self, key):
        write_int(self.stream, STATE_REQ)
        write_block(self.stream, None)
        read_int(self.stream) # flag STATE_FEED
        return read_block(self.stream)

    def set_state(self, key, binary):
        write_int(self.stream, STATE_UPDATE)
        write_block(self.stream, binary)

ENV = {
    'PYTHONPATH': os.getenv('PYTHONPATH'),
    'PY4J_SECRET': os.getenv('PY4J_SECRET'),
    'PY4J_PORT': os.getenv('PY4J_PORT'),
    'PYTHON_WORKER_FACTORY_PORT': os.getenv('PYTHON_WORKER_FACTORY_PORT'),
    'MODULE_IMPORT': os.getenv('MODULE_IMPORT'),
    'MODULE_CLASS': os.getenv('MODULE_CLASS')
}

def init_backend(custom_config, sockfile):
    backend = custom_config.get('stateBackend', 'flink')
    advance_config = custom_config.get('stateBackendAdvance', {})
    if backend == 'redis':
        print("init RedisStateBackend")
        return RedisStateBackend(advance_config)
    elif backend == 'redis-sensor':
        print("init RedisSensorStateBackend")
        return RedisSensorStateBackend(advance_config)
    else:
        print("init FlinkStateBackend")
        return FlinkStateBackend(sockfile)

def bootstrap(rad):
    boot_time = time.time()
    print(sys.version, "flink_worker start at ", boot_time)
    port = int(ENV['PY4J_PORT'])
    rpc_port = ENV['PYTHON_WORKER_FACTORY_PORT']
    params = GatewayParameters(port=port, auto_convert=True, auth_token=ENV['PY4J_SECRET'])
    gateway = JavaGateway(gateway_parameters=params)
    custom_config = gateway.entry_point.getCustomConfig()
    custom_config = javaToPython(custom_config)
    print(custom_config)
    rad.configure(custom_config)
    res = socket.getaddrinfo("127.0.0.1", rpc_port, socket.AF_UNSPEC, socket.SOCK_STREAM)
    try:
        af, socktype, proto, _, sa = res[0]
        sock = socket.socket(af, socktype, proto)
        # sock.settimeout(60)
        sock.connect(sa)
        print("connected rpc server :", rpc_port)
        sockfile = sock.makefile("rwb", 65536)
        # handshake
        handshake(sockfile, ENV['PY4J_SECRET'])
        print("finish handshake to server")
        state_backend = init_backend(custom_config, sockfile)
        while True:
            event, key, need_snapshot = read_event(sockfile)
            if not rad.contains_key(key):
                print("getting snapshot state for key", key)
                s = state_backend.get_state(key)
                rad.init_state(key, s)
            scored = rad.score(event, key)
            if need_snapshot == True and rad.contains_key(key):
                print("setting snapshot state for key", key)
                # write state back
                state_backend.set_state(key, rad.serialize_state(key))
            if isinstance(scored, list):
                write_events(sockfile, scored)
            else:
                write_events(sockfile, [scored])
    except socket.error as e:
        print(e)
        sock.close()
        gateway.close()

import importlib
print("import", ENV["MODULE_IMPORT"], "class", ENV["MODULE_CLASS"])
module = importlib.import_module(ENV["MODULE_IMPORT"])
class_ = getattr(module, ENV["MODULE_CLASS"])
bootstrap(class_())

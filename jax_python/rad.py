import json
from abc import abstractmethod

class RealtimeAnomalyDetect(object):
    """
    self.state[key] get state of the specified key
    """
    def __init__(self):
        self.state = {}

    def contains_key(self, key):
        if key is None:
            return True
        return self.state.__contains__(key)

    def get_state(self, key):
        return self.state[key]


    def init_state(self, key, binary):
        s = self.deserialize_state(binary)
        self.state[key] = s

    @abstractmethod
    def configure(self, dict):
        """
        configure the class once
        :param dic: dict that contains string-string like configuration
        :return: no need to return
        """
        pass

    @abstractmethod
    def score(self, record, key=None):
        """
        take a record, do detect, and return one or more records. `key` represents the partition of the record,
        typical implementation should get history state from key, than do detect the record
        use self.state[key](a dict to store and fetch state) to operate the state you need

        record is a dict represents the data which should be check and detect.
        :param record: the record self, dict
        :param key: key value for which group by
        :return: list of dict that represent the result of score
        the original
        """
        pass

    def serialize_state(self, key):
        """
        serialize a state by key, return binary
        :param key:
        :return: serialized binary by key in self.state
        """
        s = self.get_state(key)
        if s is not None:
            return bytes(json.dumps(s), encoding='utf-8')
        else:
            return None

    def deserialize_state(self, binary):
        """
        deserialize binary
        :return: deserialized data by binary
        """
        if binary is not None:
            return json.loads(str(binary, encoding="utf-8"))
        else:
            return None

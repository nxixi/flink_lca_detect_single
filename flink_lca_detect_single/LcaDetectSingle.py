from jax_python.rad import RealtimeAnomalyDetect
from flink_lca_detect_single import numpy_json
from flink_lca_detect_single.LcaDetect import LcaDetect

__version__="2.1"


class LcaDetectSingle(RealtimeAnomalyDetect):

	def __init__(self):
		super(LcaDetectSingle, self).__init__()
		self.config = {}

	def version(self):
		return __version__

	def configure(self, config):
		self.lca_detect = LcaDetect.from_map(config)
		self.config = config

	def score(self, record, key=None):

		if key is None:
			key = 'None'
		if self.contains_key(key) and self.state[key] is not None:
			self.lca_detect.load_state(self.state[key])
		else:
			self.lca_detect = LcaDetect.from_map(self.config)

		result = self.lca_detect.run(record)

		self.state[key] = self.lca_detect.save_state()

		return result

	def serialize_state(self, key):
		s = self.get_state(key)
		if s is not None:
			return bytes(numpy_json.dumps(s), encoding='utf-8')
		else:
			return None

	def deserialize_state(self, binary):
		if binary is not None:
			obj = numpy_json.loads(str(binary, encoding="utf-8"))
			return obj
		else:
			return None
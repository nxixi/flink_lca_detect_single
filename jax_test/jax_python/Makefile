
.PHONY: all
all: package sample

.PHONY: package
package:
	mkdir jax_python && cp __init__.py aad.py flink_worker.py rad.py shim.py spark_bridge.py jax_python
	zip -q -r jax_python.zip jax_python
	rm -rf jax_python

.PHONY: sample
sample:
	zip -q -r jax_python_sample.zip jax_python_sample jax_conf.json

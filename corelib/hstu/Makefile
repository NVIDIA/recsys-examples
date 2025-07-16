install:
	@(\
		export HSTU_DISABLE_BACKWARD=FALSE; \
		export HSTU_DISABLE_LOCAL=FALSE; \
		export HSTU_DISABLE_CAUSAL=FALSE; \
		export HSTU_DISABLE_CONTEXT=FALSE; \
		export HSTU_DISABLE_TARGET=FALSE; \
		export HSTU_DISABLE_DELTA_Q=FALSE; \
		export HSTU_DISABLE_RAB=FALSE; \
		export HSTU_DISABLE_DRAB=FALSE; \
		export HSTU_DISABLE_BF16=FALSE; \
		export HSTU_DISABLE_FP16=FALSE; \
		export HSTU_DISABLE_HDIM32=FALSE; \
		export HSTU_DISABLE_HDIM64=FALSE; \
		export HSTU_DISABLE_HDIM128=FALSE; \
		export HSTU_DISABLE_HDIM256=FALSE; \
		export HSTU_DISABLE_86OR89=TRUE; \
		python setup.py install \
	)

clean:
	rm -rf build/*
	rm -rf dist/*
	rm -rf hstu_attn.egg-info/*

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

bb:
	python3 setup.py develop --user

tt:
	python3 test.py

vt:
	pytest -q -s test.py

fm:
	ncu --set full --import-source yes --nvtx --nvtx-include "triton/" --nvtx-include  "hstu_varlen_fwd_kernel/"  -f -o hstu.%p python3 test.py

bm:
	ncu --set full --nvtx --nvtx-include "triton/" --nvtx-include "hstu_varlen_fwd_kernel/" --nvtx-include "hstu_varlen_bwd_kernel/"  -f -o hstu.%p python3 test.py

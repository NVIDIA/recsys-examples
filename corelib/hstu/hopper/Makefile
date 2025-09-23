install:
	@(\
		export HSTU_DISABLE_BACKWARD=FALSE; \
		export HSTU_DISABLE_LOCAL=FALSE; \
		export HSTU_DISABLE_CAUSAL=FALSE; \
		export HSTU_DISABLE_TARGET=FALSE; \
		export HSTU_DISABLE_CONTEXT=FALSE; \
		export HSTU_DISABLE_ARBITRARY=FALSE; \
		export HSTU_ARBITRARY_NFUNC=3; \
		export HSTU_DISABLE_RAB=FALSE; \
		export HSTU_DISABLE_DRAB=FALSE; \
		export HSTU_DISABLE_BF16=FALSE; \
		export HSTU_DISABLE_FP16=TRUE; \
		export HSTU_DISABLE_FP8=FALSE; \
		export HSTU_USE_E5M2_BWD=FALSE; \
		export HSTU_DISABLE_HDIM32=FALSE; \
		export HSTU_DISABLE_HDIM64=FALSE; \
		export HSTU_DISABLE_HDIM128=FALSE; \
		export HSTU_DISABLE_HDIM256=FALSE; \
		export HSTU_DISABLE_SM8x=TRUE; \
		python setup.py install \
	)

clean:
	rm -rf build/*
	rm -rf dist/*
	rm -rf hstu_hopper.egg-info/*
	rm -rf instantiations/hstu*

tt:
	export PYTHONPATH=${PWD}
	python3 ../test.py

vt:
	pytest -q -s ../test.py

fm:
	ncu --set full --nvtx --nvtx-include  "hstu_varlen_fwd_kernel/" --kernel-id ::regex:"compute_attn_ws|device_kernel": -f -o hstu_fwd.%p --import-source yes make tt

bm:
	ncu --set full --import-source yes --clock-control none -f -o hstu.%p --kernel-id ::regex:"compute_attn_ws|device_kernel": python ../test.py

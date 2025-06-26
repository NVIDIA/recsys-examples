from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="append_kvcache",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "append_kvcache",
            ["csrc/append_paged_kv_cache_gpu.cu", "csrc/append_paged_kv_cache.cpp"]
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
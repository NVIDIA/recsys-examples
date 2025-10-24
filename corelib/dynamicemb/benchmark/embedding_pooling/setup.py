from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='embedding_pooling_cuda',
    ext_modules=[
        CUDAExtension(
            name='embedding_pooling_cuda',
            sources=[
                'embedding_pooling_cuda.cpp',
                'embedding_pooling_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-lineinfo',
                    '--generate-line-info',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


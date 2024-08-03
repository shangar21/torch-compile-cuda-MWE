from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matDxD',
    ext_modules=[
        CUDAExtension(
            name='matDxD',
            sources=['matDxD_cuda.cu'],  # This should be the name of your C++ file
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

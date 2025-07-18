from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths
import os

os.environ.setdefault('CUDA_HOME', '/storage/software/cuda/cuda-12.1.1')
os.environ.setdefault('CC', '/storage/software/compiler/gcc-10.1.0/bin/gcc')
os.environ.setdefault('CXX', '/storage/software/compiler/gcc-10.1.0/bin/g++')

ABI = os.environ.get("TORCH_CXX11_ABI", "1")
def get_abi_flag():
    return f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"

cuda_home = os.environ['CUDA_HOME']
extra_library_dirs = library_paths() + [os.path.join(cuda_home, 'lib64')]
extra_include_dirs = include_paths() + [os.path.join(cuda_home, 'include')]

setup(
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="sam2._C",
            sources=["src/sam2/csrc/connected_components.cu"],
            include_dirs=extra_include_dirs,
            library_dirs=extra_library_dirs,
            libraries=[
                "c10", "torch", "torch_cpu", "torch_cuda",
                "c10_cuda", "torch_python",
                "cusparse", "cudnn", "nvJitLink"
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    get_abi_flag(),
                    "-std=c++17"
                ],
                "nvcc": [
                    "-O3",
                    get_abi_flag(),
                    "-std=c++17",
                    "-Xcompiler", get_abi_flag()
                ]
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)

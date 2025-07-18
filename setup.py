from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths

import os
ABI = os.environ.get("TORCH_CXX11_ABI", "1")

def get_abi_flag():
    return f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"

setup(
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="sam2._C",
            sources=["src/sam2/csrc/connected_components.cu"],
            include_dirs=include_paths(),
            library_dirs=library_paths(),
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
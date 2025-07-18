from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, include_paths, library_paths

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
                "cxx": ["-O3", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": ["-O3", "-Xcompiler", "-D_GLIBCXX_USE_CXX11_ABI=0"]
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
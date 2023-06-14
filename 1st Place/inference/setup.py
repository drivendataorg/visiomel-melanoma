# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, os
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from setuptools import setup, find_packages

if torch.cuda.is_available():
    assert torch.matmul(torch.ones(2097153,2).cuda(),torch.ones(2,2).cuda()).min().item()==2, 'Please upgrade from CUDA 9.0 to CUDA 10.0+'

this_dir = os.path.dirname(os.path.realpath(__file__))
torch_dir = os.path.dirname(torch.__file__)
conda_include_dir = '/'.join(torch_dir.split('/')[:-4]) + '/include'
extra = {'cxx': ['-std=c++14', '-fopenmp','-O3'], 'nvcc': ['-std=c++14', '-Xcompiler', '-fopenmp', '-O3']}

setup(
    name='sparseconvnet',
    version='0.2',
    description='Submanifold (Spatially) Sparse Convolutional Networks https://arxiv.org/abs/1706.01307',
    author='Facebook AI Research',
    author_email='benjamingraham@fb.com',
    url='https://github.com/facebookresearch/SparseConvNet',
    packages=['sparseconvnet','sparseconvnet.SCN'],
    ext_modules=[
      CppExtension('sparseconvnet.SCN',
        ['sparseconvnet/SCN/pybind.cpp', 'sparseconvnet/SCN/sparseconvnet_cpu.cpp'],
        include_dirs=[this_dir+'/sparseconvnet/SCN/'],
        extra_compile_args=extra['cxx'])],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)

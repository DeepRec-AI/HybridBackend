# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

r'''HybridBackend setup script.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

from hybridbackend import __version__
from hybridbackend import __author__

PACKAGES = find_packages(exclude=['cpp', 'tests', 'examples'])
PACKAGE_DATA = {'': ['*.so', '*.so.*']}

try:
  from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

  class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
      _bdist_wheel.finalize_options(self)
      self.root_is_pure = False

    def get_tag(self):
      python, abi, plat = _bdist_wheel.get_tag(self)
      plat = 'manylinux1_x86_64'
      return python, abi, plat
except ImportError:
  bdist_wheel = None


class BinaryDistribution(Distribution):
  r'''This class is needed in order to create OS specific wheels.
  '''

  def has_ext_modules(self):
    return True

setup(
    name='hybridbackend',
    version=__version__,
    packages=PACKAGES,
    include_package_data=True,
    package_data=PACKAGE_DATA,
    install_requires=[],
    cmdclass={'bdist_wheel': bdist_wheel},
    distclass=BinaryDistribution,
    zip_safe=False,
    author=__author__,
    description='Efficient training of deep recommenders on cloud.',
    long_description=(
        "HybridBackend is a training framework for deep recommenders which "
        "bridges gap between evolving cloud infrastructure and complex "
        "training process."),
    long_description_content_type='text/markdown',
    url="https://github.com/alibaba/HybridBackend",
    project_urls={
        'Bug Tracker': 'https://github.com/alibaba/HybridBackend/issues',
        'Documentation': 'https://hybridbackend.readthedocs.io/en/latest/',
        'Source Code': 'https://github.com/alibaba/HybridBackend',
    },
    keywords=('deep learning', 'recommendation system'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache License 2.0',
    license_files=('LICENSE', 'NOTICE'),
)

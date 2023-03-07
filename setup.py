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

import os

from Cython.Distutils import build_ext
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.extension import Extension

from hybridbackend import __author__
from hybridbackend import __version__

NAME = f'hybridbackend{os.getenv("WHEEL_ALIAS", "")}'
VERSION = f'{__version__}{os.getenv("WHEEL_BUILD", "")}'
REQUIRES = os.getenv('WHEEL_REQUIRES', '').split(';')
PACKAGES = find_packages(exclude=['tests'])
PACKAGE_DATA = {'': ['*.so', '*.so.*']}


class NoExtensionBuilder(build_ext):
  r'''Build extensions to do nothing.
  '''
  def build_extension(self, ext):
    return


CMDCLASS = dict(build_ext=NoExtensionBuilder)
DESCRIPTION = ('A high-performance framework for training wide-and-deep '
               'recommender systems on heterogeneous cluster')
LONG_DESCRIPTION = DESCRIPTION
with open(
    os.path.join(os.path.dirname(__file__), 'README.md'),
    encoding='UTF-8') as readme:
  LONG_DESCRIPTION = readme.read()


class BinaryDistribution(Distribution):
  r'''This class is needed in order to create OS specific wheels.
  '''
  def has_ext_modules(self):
    return True

  def is_pure(self):
    return False


setup(
  name=NAME,
  version=VERSION,
  packages=PACKAGES,
  include_package_data=True,
  package_data=PACKAGE_DATA,
  install_requires=REQUIRES,
  ext_modules=[Extension('', sources=[])],
  cmdclass=CMDCLASS,
  distclass=BinaryDistribution,
  zip_safe=False,
  author=__author__,
  description=DESCRIPTION,
  long_description=LONG_DESCRIPTION,
  long_description_content_type='text/markdown',
  url='https://github.com/alibaba/HybridBackend',
  download_url='https://github.com/alibaba/HybridBackend/tags',
  project_urls={
    'Bug Tracker': 'https://github.com/alibaba/HybridBackend/issues',
    'Documentation': 'https://hybridbackend.readthedocs.io/en/latest/',
    'Source Code': 'https://github.com/alibaba/HybridBackend',
  },
  keywords=['deep learning', 'recommendation system'],
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

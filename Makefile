LIBNAME := hybridbackend
OS ?= $(shell uname -s)
PROCESSOR_ARCHITECTURE ?= $(shell uname -p)

HYBRIDBACKEND_WITH_BUILDINFO ?= ON
HYBRIDBACKEND_WITH_CUDA ?= ON
HYBRIDBACKEND_WITH_CUDA_GENCODE ?= "70 75 80 86"
HYBRIDBACKEND_WITH_NVTX ?= ON
HYBRIDBACKEND_WITH_NCCL ?= ON
HYBRIDBACKEND_WITH_ARROW ?= ON
HYBRIDBACKEND_WITH_ARROW_ZEROCOPY ?= ON
HYBRIDBACKEND_WITH_ARROW_HDFS ?= ON
HYBRIDBACKEND_WITH_ARROW_S3 ?= ON
HYBRIDBACKEND_WITH_SPARSEHASH ?= ON
HYBRIDBACKEND_WITH_TENSORFLOW ?= ON
HYBRIDBACKEND_USE_CXX11_ABI ?= 0
HYBRIDBACKEND_DEBUG ?= OFF
HYBRIDBACKEND_WHEEL_ALIAS ?= ""
HYBRIDBACKEND_WHEEL_BUILD ?= ""
HYBRIDBACKEND_WHEEL_REQUIRES ?= ""
HYBRIDBACKEND_WHEEL_DEBUG ?= OFF
HYBRIDBACKEND_CHECK_INSTANCE ?= OFF

CXX ?= gcc
PYTHON ?= python

CFLAGS := \
	-D_GLIBCXX_USE_CXX11_ABI=$(HYBRIDBACKEND_USE_CXX11_ABI) \
	-I$(LIBNAME)/include \
	-I.

CXX_CFLAGS := -std=c++11 \
	-fstack-protector \
	-Wall \
	-Werror \
	-Wno-sign-compare \
	-Wformat \
	-Wformat-security

LDFLAGS := -shared \
	-fstack-protector \
	-fpic

ifeq ($(HYBRIDBACKEND_DEBUG),ON)
CFLAGS := $(CFLAGS) -g -O0
else
CFLAGS := $(CFLAGS) -g -O3 -DNDEBUG
endif

ifeq ($(OS),Darwin)
OSX_TARGET ?= $(shell sw_vers -productVersion)
PYTHON_HOME ?= /usr/local
PYTHON_IMPL ?=
PYTHON_IMPL_FLAG ?=
CFLAGS := $(CFLAGS) \
	-isystem $(PYTHON_HOME)/include/$(PYTHON_IMPL)$(PYTHON_IMPL_FLAG) \
	-mmacosx-version-min=$(OSX_TARGET)

LDFLAGS := $(LDFLAGS) \
	-L$(PYTHON_HOME)/lib -l$(PYTHON_IMPL)
endif

ifeq ($(HYBRIDBACKEND_WITH_BUILDINFO),ON)
HYBRIDBACKEND_BUILD_VERSION := $(shell \
	grep __version__ hybridbackend/__init__.py | cut -d\' -f2 \
	2>/dev/null)
HYBRIDBACKEND_BUILD_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
HYBRIDBACKEND_BUILD_GLIBC := $(shell \
	ldd --version | head -1 | rev | cut -d' ' -f1 | rev 2>/dev/null)
HYBRIDBACKEND_BUILD_CXX := $(shell \
	$(CXX) --version | head -1 | rev | cut -d ' ' -f1 | rev \
	2>/dev/null)

CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_BUILDINFO=1 \
	-DHYBRIDBACKEND_BUILD_VERSION="\"$(HYBRIDBACKEND_BUILD_VERSION)\"" \
	-DHYBRIDBACKEND_BUILD_COMMIT="\"$(HYBRIDBACKEND_BUILD_COMMIT)\"" \
	-DHYBRIDBACKEND_BUILD_GLIBC="\"$(HYBRIDBACKEND_BUILD_GLIBC)\"" \
	-DHYBRIDBACKEND_BUILD_CXX="\"$(HYBRIDBACKEND_BUILD_CXX)\"" \
	-DHYBRIDBACKEND_BUILD_CXX11_ABI="\"$(HYBRIDBACKEND_USE_CXX11_ABI)\""
endif

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
HYBRIDBACKEND_CUDA_GENCODE := $(shell \
	echo $(HYBRIDBACKEND_WITH_CUDA_GENCODE) | tr ' ' ',' \
	2>/dev/null)
HYBRIDBACKEND_CUDA_CC := $(shell \
	$(NVCC) --version | tail -1 | cut -d' ' -f2 \
	2>/dev/null)
CFLAGS := $(CFLAGS) \
	-isystem $(CUDA_HOME)/include \
	-DHYBRIDBACKEND_CUDA=1 \
	-DHYBRIDBACKEND_CUDA_GENCODE="\"$(HYBRIDBACKEND_CUDA_GENCODE)\"" \
	-DHYBRIDBACKEND_CUDA_CC="\"$(HYBRIDBACKEND_CUDA_CC)\""
NVCC_CFLAGS := --std=c++11 \
	--expt-relaxed-constexpr \
	--expt-extended-lambda \
	--disable-warnings \
	$(foreach cc, $(HYBRIDBACKEND_WITH_CUDA_GENCODE),\
	 -gencode arch=compute_$(cc),code=sm_$(cc))
LDFLAGS := $(LDFLAGS) \
	-L$(CUDA_HOME)/lib64 \
	-lcudart
ifeq ($(HYBRIDBACKEND_WITH_NVTX),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_NVTX=1
LDFLAGS := $(LDFLAGS) -lnvToolsExt
endif
ifeq ($(HYBRIDBACKEND_WITH_NCCL),ON)
NCCL_HOME ?= /usr/local
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_NCCL=1 \
	-isystem $(NCCL_HOME)/include
LDFLAGS := $(LDFLAGS) \
	-L$(NCCL_HOME)/lib64 \
	-L$(NCCL_HOME)/lib \
	-lnccl
endif
endif

ifeq ($(HYBRIDBACKEND_CHECK_INSTANCE),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_CHECK_INSTANCE=1
endif

ifneq ($(OS),Darwin)
D_FILES := $(shell \
    find \( -path ./cibuild -o -path ./build -o -path ./dist \) \
	-prune -false -o -type f -name '*.d' \
	-exec realpath {} --relative-to . \;)

-include $(D_FILES)
endif

THIRDPARTY_DEPS :=
ifeq ($(HYBRIDBACKEND_WITH_ARROW),ON)
ARROW_HOME ?= cibuild/arrow/dist
ARROW_API_H := $(ARROW_HOME)/include/arrow/api.h
THIRDPARTY_DEPS := $(THIRDPARTY_DEPS) $(ARROW_API_H)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_ARROW=1 \
	-isystem $(ARROW_HOME)/include
ifeq ($(HYBRIDBACKEND_WITH_ARROW_ZEROCOPY),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_ARROW_ZEROCOPY=1
endif
ifeq ($(HYBRIDBACKEND_WITH_ARROW_HDFS),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_ARROW_HDFS=1
endif
ifeq ($(HYBRIDBACKEND_WITH_ARROW_S3),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_ARROW_S3=1
endif
COMMON_LDFLAGS := \
	-L$(ARROW_HOME)/lib \
	-larrow \
	-larrow_dataset \
	-larrow_bundled_dependencies \
	-lparquet \
	-lcurl
ifneq ($(OS),Darwin)
COMMON_LDFLAGS := \
	-Bsymbolic \
	-Wl,--whole-archive \
	$(COMMON_LDFLAGS) \
	-Wl,--no-whole-archive
endif
RE2_HOME ?=
ifneq ($(strip $(RE2_HOME)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(RE2_HOME)/lib -lre2
endif
THRIFT_HOME ?=
ifneq ($(strip $(THRIFT_HOME)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(THRIFT_HOME)/lib -lthrift
endif
UTF8PROC_HOME ?=
ifneq ($(strip $(UTF8PROC_HOME)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(UTF8PROC_HOME)/lib -lutf8proc
endif
SNAPPY_HOME ?=
ifneq ($(strip $(SNAPPY_HOME)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(SNAPPY_HOME)/lib -lsnappy
endif
ZSTD_HOME ?=
ifneq ($(strip $(ZSTD_HOME)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(ZSTD_HOME)/lib -lzstd
endif
ZLIB_HOME ?=
ifneq ($(strip $(ZLIB_HOME)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(ZLIB_HOME)/lib -lz
endif
SSL_HOME ?= /usr/local
COMMON_LDFLAGS := $(COMMON_LDFLAGS) \
	-L$(SSL_HOME)/lib \
	-lssl \
	-lcrypto
endif

ifeq ($(HYBRIDBACKEND_WITH_SPARSEHASH),ON)
SPARSEHASH_HOME ?= cibuild/sparsehash/dist
SPARSEHASH_DENSE_HASH_MAP := $(SPARSEHASH_HOME)/include/sparsehash/dense_hash_map
THIRDPARTY_DEPS := $(THIRDPARTY_DEPS) $(SPARSEHASH_DENSE_HASH_MAP)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_SPARSEHASH=1 \
	-isystem ${SPARSEHASH_HOME}/include
LDFLAGS := $(LDFLAGS) \
	-lpthread
endif

COMMON_LIB := $(LIBNAME)/lib$(LIBNAME).so
-include $(LIBNAME)/cpp/common/Makefile
CORE_DEPS := $(COMMON_LIB)

ifeq ($(HYBRIDBACKEND_WITH_TENSORFLOW),ON)
TENSORFLOW_LIB := $(LIBNAME)/tensorflow/lib$(LIBNAME)_tensorflow.so
-include $(LIBNAME)/cpp/tensorflow/Makefile
CORE_DEPS := $(CORE_DEPS) $(TENSORFLOW_LIB)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_TENSORFLOW=1
endif

.PHONY: build
build: $(CORE_DEPS)
	WHEEL_ALIAS="$(HYBRIDBACKEND_WHEEL_ALIAS)" \
	WHEEL_BUILD="$(HYBRIDBACKEND_WHEEL_BUILD)" \
	WHEEL_REQUIRES="$(HYBRIDBACKEND_WHEEL_REQUIRES)" \
	WHEEL_DEBUG="$(HYBRIDBACKEND_WHEEL_DEBUG)" \
	$(PYTHON) setup.py bdist_wheel -d cibuild/dist
	@ls cibuild/dist/*.whl

TESTS := $(shell find hybridbackend/ -type f -name "*_test.py")

.PHONY: test
test:
	for t in $(TESTS); do \
		echo -e "\033[1;33m[TEST] $$t \033[0m" ; \
		$(PYTHON) $$t || exit 1; \
		echo ; \
	done

.PHONY: doc
doc:
	sphinx-build -M html docs/ cibuild/dist/doc

.PHONY: lint
lint:
	cibuild/lint

.PHONY: cibuild
cibuild: lint build doc test

.PHONY: clean
clean:
	rm -fr cibuild/dist/
	rm -fr build/
	rm -fr *.egg-info/
	rm -rf .pylint.d/
	find -name *.c -exec rm -fr {} \;
	find -name *.o -exec rm -fr {} \;
	find -name *.d -exec rm -fr {} \;
	find -name *.so -exec rm -fr {} \;

.DEFAULT_GOAL := build

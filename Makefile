LIBNAME := hybridbackend
OS ?= $(shell uname -s)
PROCESSOR_ARCHITECTURE ?= $(shell uname -p)

HYBRIDBACKEND_CONFIG=$(wildcard .config.mk)
include $(HYBRIDBACKEND_CONFIG)

HYBRIDBACKEND_WITH_CUDA ?= ON
HYBRIDBACKEND_WITH_CUDA_GENCODE ?= 70 75 80 86
HYBRIDBACKEND_WITH_NVTX ?= ON
HYBRIDBACKEND_WITH_NCCL ?= ON
HYBRIDBACKEND_WITH_ARROW ?= ON
HYBRIDBACKEND_WITH_ARROW_ZEROCOPY ?= ON
HYBRIDBACKEND_WITH_ARROW_HDFS ?= ON
HYBRIDBACKEND_WITH_ARROW_S3 ?= ON
HYBRIDBACKEND_WITH_TENSORFLOW ?= ON
HYBRIDBACKEND_WITH_TENSORFLOW_ESTIMATOR ?= ON
HYBRIDBACKEND_WITH_TENSORFLOW_HALF ?= ON
HYBRIDBACKEND_WITH_TENSORFLOW_DISTRO ?= 1015
HYBRIDBACKEND_WITH_BUILDINFO ?= ON
HYBRIDBACKEND_USE_CXX11_ABI ?= 0
HYBRIDBACKEND_DEBUG ?= OFF
HYBRIDBACKEND_WHEEL_ALIAS ?=
HYBRIDBACKEND_WHEEL_BUILD ?=
HYBRIDBACKEND_WHEEL_REQUIRES ?=
HYBRIDBACKEND_WHEEL_REPAIR ?= ON
HYBRIDBACKEND_WHEEL_POSTCHECK ?= ON
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
PYTHON_INCLUDE ?= /usr/local/include
PYTHON_LIB ?= /usr/local/lib
PYTHON_IMPL ?=
PYTHON_IMPL_FLAG ?=
CFLAGS := $(CFLAGS) \
	-isystem $(PYTHON_INCLUDE)/$(PYTHON_IMPL)$(PYTHON_IMPL_FLAG) \
	-mmacosx-version-min=$(OSX_TARGET)

LDFLAGS := $(LDFLAGS) -L$(PYTHON_LIB) -l$(PYTHON_IMPL)
endif

ifeq ($(HYBRIDBACKEND_WITH_BUILDINFO),ON)
HYBRIDBACKEND_BUILD_VERSION := $(shell \
	grep __version__ hybridbackend/__init__.py | cut -d\' -f2 \
	2>/dev/null)
HYBRIDBACKEND_BUILD_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
HYBRIDBACKEND_BUILD_LOG := $(shell \
	git log -5 --date=short --pretty='%cd:%h' | xargs | tr ' ' ',' \
	2>/dev/null)
HYBRIDBACKEND_BUILD_CXX := $(shell \
	$(CXX) --version | head -1 | rev | cut -d ' ' -f1,4 | rev \
	2>/dev/null)

CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_BUILDINFO=1 \
	-DHYBRIDBACKEND_BUILD_VERSION="\"$(HYBRIDBACKEND_BUILD_VERSION)\"" \
	-DHYBRIDBACKEND_BUILD_COMMIT="\"$(HYBRIDBACKEND_BUILD_COMMIT)\"" \
	-DHYBRIDBACKEND_BUILD_LOG="\"$(HYBRIDBACKEND_BUILD_LOG)\"" \
	-DHYBRIDBACKEND_BUILD_CXX="\"$(HYBRIDBACKEND_BUILD_CXX)\"" \
	-DHYBRIDBACKEND_BUILD_CXX11_ABI=$(HYBRIDBACKEND_USE_CXX11_ABI)
endif

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
NVCC ?= nvcc
CUDA_INCLUDE ?= /usr/local/cuda/include
CUDA_LIB ?= /usr/local/cuda/lib64
HYBRIDBACKEND_CUDA_GENCODE := $(shell \
	echo "$(HYBRIDBACKEND_WITH_CUDA_GENCODE)" | tr ' ' ',' \
	2>/dev/null)
CFLAGS := $(CFLAGS) \
	-isystem $(CUDA_INCLUDE) \
	-DHYBRIDBACKEND_CUDA=1 \
	-DHYBRIDBACKEND_CUDA_GENCODE="\"$(HYBRIDBACKEND_CUDA_GENCODE)\""
NVCC_CFLAGS := --std=c++11 \
	-lineinfo \
	--expt-relaxed-constexpr \
	--expt-extended-lambda \
	--disable-warnings \
	$(foreach cc, $(HYBRIDBACKEND_WITH_CUDA_GENCODE),\
	 -gencode arch=compute_$(cc),code=sm_$(cc))
LDFLAGS := $(LDFLAGS) \
	-L$(CUDA_LIB) \
	-lcudart
ifeq ($(HYBRIDBACKEND_WITH_NVTX),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_NVTX=1
LDFLAGS := $(LDFLAGS) -lnvToolsExt
endif
ifeq ($(HYBRIDBACKEND_WITH_NCCL),ON)
NCCL_INCLUDE ?= /usr/local/nccl/include
NCCL_LIB ?= /usr/local/nccl/lib
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_NCCL=1 \
	-isystem $(NCCL_INCLUDE)
LDFLAGS := $(LDFLAGS) \
	-L$(NCCL_LIB) \
	-lnccl
endif
endif

ifeq ($(HYBRIDBACKEND_CHECK_INSTANCE),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_CHECK_INSTANCE=1
endif

ifneq ($(OS),Darwin)
D_FILES := $(shell \
    find \( -path ./build \) \
	-prune -false -o -type f -name '*.d' \
	-exec realpath {} --relative-to . \;)

-include $(D_FILES)
endif

THIRDPARTY_DEPS :=
SSL_LIB ?= /usr/lib/x86_64-linux-gnu

COMMON_LDFLAGS := $(COMMON_LDFLAGS) \
	-Bsymbolic \
	-L$(SSL_LIB) \
	-lssl \
	-lcrypto \
	-lcurl
ifeq ($(HYBRIDBACKEND_WITH_ARROW),ON)
ARROW_INCLUDE ?= build/arrow/dist/include
ARROW_LIB ?= build/arrow/dist/lib
ARROW_API_H := $(ARROW_INCLUDE)/arrow/api.h
THIRDPARTY_DEPS := $(THIRDPARTY_DEPS) $(ARROW_API_H)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_ARROW=1 \
	-isystem $(ARROW_INCLUDE)
ifeq ($(HYBRIDBACKEND_WITH_ARROW_ZEROCOPY),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_ARROW_ZEROCOPY=1
endif
ifeq ($(HYBRIDBACKEND_WITH_ARROW_HDFS),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_ARROW_HDFS=1
endif
ifeq ($(HYBRIDBACKEND_WITH_ARROW_S3),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_ARROW_S3=1
endif
ifeq ($(OS),Darwin)
COMMON_LDFLAGS := \
	$(COMMON_LDFLAGS) \
	-L$(ARROW_LIB) \
	-larrow \
	-larrow_dataset \
	-larrow_bundled_dependencies \
	-lparquet
else
COMMON_LDFLAGS := \
	$(COMMON_LDFLAGS) \
	-Wl,--whole-archive \
	-L$(ARROW_LIB) \
	-larrow \
	-larrow_dataset \
	-larrow_bundled_dependencies \
	-lparquet \
	-Wl,--no-whole-archive
endif
RE2_LIB ?=
ifneq ($(strip $(RE2_LIB)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(RE2_LIB) -lre2
endif
THRIFT_LIB ?=
ifneq ($(strip $(THRIFT_LIB)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(THRIFT_LIB) -lthrift
endif
UTF8PROC_LIB ?=
ifneq ($(strip $(UTF8PROC_LIB)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(UTF8PROC_LIB) -lutf8proc
endif
LZ4_LIB ?=
ifneq ($(strip $(LZ4_LIB)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(LZ4_LIB) -llz4
endif
SNAPPY_LIB ?=
ifneq ($(strip $(SNAPPY_LIB)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(SNAPPY_LIB) -lsnappy
endif
ZSTD_LIB ?=
ifneq ($(strip $(ZSTD_LIB)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(ZSTD_LIB) -lzstd
endif
ZLIB_LIB ?=
ifneq ($(strip $(ZLIB_LIB)),)
COMMON_LDFLAGS := $(COMMON_LDFLAGS) -L$(ZLIB_LIB) -lz
endif
endif

COMMON_LIB := $(LIBNAME)/lib$(LIBNAME).so
-include $(LIBNAME)/common/Makefile
CORE_DEPS := $(COMMON_LIB)

ifeq ($(HYBRIDBACKEND_WITH_TENSORFLOW),ON)
TENSORFLOW_LIB := $(LIBNAME)/tensorflow/lib$(LIBNAME)_tensorflow.so
-include $(LIBNAME)/tensorflow/Makefile
CORE_DEPS := $(CORE_DEPS) $(TENSORFLOW_LIB)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_TENSORFLOW=1
ifeq ($(HYBRIDBACKEND_WITH_TENSORFLOW_HALF),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_TENSORFLOW_HALF=1
endif
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_TENSORFLOW_DISTRO=$(HYBRIDBACKEND_WITH_TENSORFLOW_DISTRO)

TENSORFLOW_INCLUDE ?=
ifneq ($(strip $(TENSORFLOW_INCLUDE)),)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_TENSORFLOW_INTERNAL=1 \
	-isystem $(TENSORFLOW_INCLUDE)
endif
endif

.PHONY: build
build: $(CORE_DEPS)
	WHEEL_ALIAS="$(HYBRIDBACKEND_WHEEL_ALIAS)" \
	WHEEL_BUILD="$(HYBRIDBACKEND_WHEEL_BUILD)" \
	WHEEL_REQUIRES="$(HYBRIDBACKEND_WHEEL_REQUIRES)" \
	$(PYTHON) setup.py bdist_wheel -d build/wheel
	@ls build/wheel/*.whl

.PHONY: doc
doc:
	mkdir -p build/doc
	sphinx-build -M html docs/ build/doc

TESTS := $(shell find hybridbackend/ -type f -name "*_test.py")

.PHONY: test
test:
	for t in $(TESTS); do \
		echo -e "\033[1;33m[TEST] $$t \033[0m" ; \
		$(PYTHON) $$t || exit 1; \
		echo ; \
	done

.PHONY: install
ifeq ($(HYBRIDBACKEND_WHEEL_REPAIR),ON)
ifeq ($(HYBRIDBACKEND_WHEEL_POSTCHECK),ON)
install: build
	build/repair build/wheel build/release
	PYTHONPATH= pip install -U build/release/*.whl
	$(MAKE) test
	@ls build/release/*.whl
else
install: build
	build/repair build/wheel build/release
	PYTHONPATH= pip install -U build/release/*.whl
	@ls build/release/*.whl
endif
else
ifeq ($(HYBRIDBACKEND_WHEEL_POSTCHECK),ON)
install: build
	mkdir -p build/release
	cp -rf build/wheel/* build/release/
	PYTHONPATH= pip install -U build/release/*.whl
	$(MAKE) test
	@ls build/release/*.whl
else
install: build
	mkdir -p build/release
	cp -rf build/wheel/* build/release/
	PYTHONPATH= pip install -U build/release/*.whl
	@ls build/release/*.whl
endif
endif

.PHONY: clean
clean:
	rm -fr build/doc/
	rm -fr build/reports/
	rm -fr build/wheel/
	rm -fr build/release/
	rm -fr build/lib.*
	rm -fr build/bdist.*
	rm -fr build/temp.*
	rm -fr *.egg-info/
	rm -rf .pylint.d/
	find -name *.c -exec rm -fr {} \;
	find -name *.o -exec rm -fr {} \;
	find -name *.d -exec rm -fr {} \;
	find -name *.so -exec rm -fr {} \;

.DEFAULT_GOAL := build

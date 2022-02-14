LIBNAME := hybridbackend
OS ?= $(shell uname -s)
PROCESSOR_ARCHITECTURE ?= $(shell uname -p)

HYBRIDBACKEND_WITH_BUILDINFO ?= ON
HYBRIDBACKEND_WITH_CUDA ?= ON
HYBRIDBACKEND_WITH_CUDA_GENCODE ?= "70 75 86"
HYBRIDBACKEND_WITH_NCCL ?= ON
HYBRIDBACKEND_WITH_CUB ?= ON
HYBRIDBACKEND_WITH_ARROW ?= ON
HYBRIDBACKEND_WITH_ARROW_ZEROCOPY ?= ON
HYBRIDBACKEND_WITH_ARROW_HDFS ?= ON
HYBRIDBACKEND_WITH_ARROW_S3 ?= ON
HYBRIDBACKEND_WITH_ARROW_SIMD_LEVEL ?= AVX2
HYBRIDBACKEND_WITH_TENSORFLOW ?= ON
HYBRIDBACKEND_USE_CXX11_ABI ?= 0
HYBRIDBACKEND_WHEEL_ALIAS ?= ""
HYBRIDBACKEND_WHEEL_BUILD ?= ""
HYBRIDBACKEND_WHEEL_REQUIRES ?= ""

CXX ?= gcc
PYTHON ?= python
SSL_HOME ?= /usr/local

CFLAGS := -O3 -g \
	-DNDEBUG \
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
	-fpic \
	-L/usr/local \
	-L$(SSL_HOME)/lib \
	-lssl \
	-lcrypto

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
HYBRIDBACKEND_BUILD_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_BUILDINFO=1 \
	-DHYBRIDBACKEND_BUILD_COMMIT="\"$(HYBRIDBACKEND_BUILD_COMMIT)\""
endif

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CFLAGS := $(CFLAGS) \
	-isystem $(CUDA_HOME)/include
NVCC_CFLAGS := --std=c++11 \
	--expt-relaxed-constexpr \
	--expt-extended-lambda \
	--disable-warnings \
	$(foreach cc, $(HYBRIDBACKEND_WITH_CUDA_GENCODE),\
	 -gencode arch=compute_$(cc),code=sm_$(cc))
LDFLAGS := $(LDFLAGS) \
	-L$(CUDA_HOME)/lib64 \
	-lcudart
ifeq ($(HYBRIDBACKEND_WITH_CUB),ON)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_CUB=1
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

ifneq ($(OS),Darwin)
D_FILES := $(shell \
    find \( -path ./arrow -o -path ./build -o -path ./dist \) \
	-prune -false -o -type f -name '*.d' \
	-exec realpath {} --relative-to . \;)

-include $(D_FILES)
endif

THIRDPARTY_DEPS :=
ifeq ($(HYBRIDBACKEND_WITH_ARROW),ON)
include arrow/Makefile
THIRDPARTY_DEPS := $(THIRDPARTY_DEPS) $(ARROW_LIB)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_ARROW=1 \
	-isystem $(ARROW_DISTDIR)/include
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
RE2_HOME ?= /usr/local
THRIFT_HOME ?= /usr/local
UTF8PROC_HOME ?= /usr/local
SNAPPY_HOME ?= /usr/local
ZSTD_HOME ?= /usr/local
ZLIB_HOME ?= /usr/local

COMMON_LDFLAGS := \
	-L$(ARROW_DISTDIR)/lib \
	-larrow \
	-larrow_dataset \
	-larrow_bundled_dependencies \
	-lparquet \
	-lcurl \
	-L$(RE2_HOME)/lib -lre2 \
	-L$(THRIFT_HOME)/lib -lthrift \
	-L$(UTF8PROC_HOME)/lib -lutf8proc \
	-L$(SNAPPY_HOME)/lib -lsnappy \
	-L$(ZSTD_HOME)/lib -lzstd \
	-L$(ZLIB_HOME)/lib -lz
else
COMMON_LDFLAGS := \
	-L$(ARROW_DISTDIR)/lib \
	-Wl,--whole-archive \
	-larrow \
	-larrow_dataset \
	-lparquet \
	-Wl,--no-whole-archive \
	-larrow_bundled_dependencies \
	-lcurl
endif
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
	$(PYTHON) setup.py bdist_wheel -d cibuild/dist
	@ls cibuild/dist/*.whl

TESTS := $(shell \
	find tests/ -type f -name "*_test.py" \
	-exec realpath {} --relative-to . \;)

.PHONY: test
test:
	for t in $(TESTS); do \
		echo -e "\033[1;33m[TEST] $$t \033[0m" ; \
		$(PYTHON) $$t || exit 1; \
		echo ; \
	done

CPU_TESTS := $(shell \
	find tests/tensorflow/data/ -type f -name "*_test.py" \
	-exec realpath {} --relative-to . \;)

.PHONY: cpu_test
cpu_test:
	for t in $(CPU_TESTS); do \
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
	@rm -fr cibuild/dist/
	@rm -fr build/
	@rm -fr *.egg-info/
	@find -name *.o -exec rm -fr {} \;
	@find -name *.d -exec rm -fr {} \;
	@find -name *.so -exec rm -fr {} \;

.DEFAULT_GOAL := build

LIBNAME := hybridbackend

HYBRIDBACKEND_WITH_BUILDINFO ?= ON
HYBRIDBACKEND_WITH_CUDA ?= ON
HYBRIDBACKEND_WITH_CUDA_GENCODE ?= "70 75 86"
HYBRIDBACKEND_WITH_NCCL ?= ON
HYBRIDBACKEND_WITH_CUB ?= ON
HYBRIDBACKEND_WITH_GRENDEZ ?= ON
HYBRIDBACKEND_WITH_ARROW ?= ON
HYBRIDBACKEND_WITH_ARROW_ZEROCOPY ?= ON
HYBRIDBACKEND_WITH_ARROW_HDFS ?= ON
HYBRIDBACKEND_WITH_ARROW_S3 ?= ON
HYBRIDBACKEND_WITH_ARROW_SIMD_LEVEL ?= AVX512
HYBRIDBACKEND_WITH_TENSORFLOW ?= ON
HYBRIDBACKEND_USE_CXX11_ABI ?= 1

CXX ?= gcc
LOCAL_HOME ?= /usr/local
PAI_HOME ?= /home/pai

CFLAGS := -O3 -g \
	-DNDEBUG \
	-D_GLIBCXX_USE_CXX11_ABI=$(HYBRIDBACKEND_USE_CXX11_ABI) \
	-isystem $(LOCAL_HOME) \
	-isystem $(PAI_HOME)/include \
	-Ihybridbackend/include \
	-I.

CXX_CFLAGS := -std=c++11 \
	-fstack-protector \
	-Wall \
	-Werror \
	-Wno-sign-compare \
	-Wformat \
	-Wformat-security

LDFLAGS := -shared \
	-znoexecstack \
	-zrelro \
	-znow \
	-fstack-protector

COMMON_LDFLAGS := -L$(PAI_HOME)/lib

ifeq ($(HYBRIDBACKEND_WITH_BUILDINFO),ON)
HYBRIDBACKEND_BUILD_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_BUILDINFO=1 \
	-DHYBRIDBACKEND_BUILD_COMMIT="\"$(HYBRIDBACKEND_BUILD_COMMIT)\""
endif

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
NVCC ?= nvcc
CUDA_HOME ?= $(LOCAL_HOME)/cuda
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
NCCL_HOME ?= $(PAI_HOME)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_NCCL=1 \
	-isystem $(NCCL_HOME)/include
LDFLAGS := $(LDFLAGS) \
	-L$(NCCL_HOME)/lib64 \
	-L$(NCCL_HOME)/lib \
	-lnccl
endif
endif
ifeq ($(HYBRIDBACKEND_WITH_GRENDEZ),ON)
CFLAGS := $(CFLAGS) -DHYBRIDBACKEND_GRENDEZ=1
endif

D_FILES := $(shell \
    find \( -path ./arrow -o -path ./build -o -path ./dist \) \
	-prune -false -o -type f -name '*.d' \
	-exec realpath {} --relative-to . \;)

-include $(D_FILES)

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
COMMON_LDFLAGS := $(COMMON_LDFLAGS) \
        -L$(ARROW_DISTDIR)/lib \
	-Wl,--whole-archive \
	-larrow \
	-larrow_dataset \
	-lparquet \
	-Wl,--no-whole-archive \
	-larrow_bundled_dependencies
endif

-include hybridbackend/cpp/common/Makefile
CORE_DEPS := $(COMMON_LIB)

ifeq ($(HYBRIDBACKEND_WITH_TENSORFLOW),ON)
TENSORFLOW_LIB := hybridbackend/tensorflow/lib$(LIBNAME)_tensorflow.so
-include hybridbackend/cpp/tensorflow/Makefile
CORE_DEPS := $(CORE_DEPS) $(TENSORFLOW_LIB)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_TENSORFLOW=1
endif

.PHONY: build
build: $(CORE_DEPS)
	python setup.py bdist_wheel
	@ls dist/*.whl

TESTS := $(shell \
	find tests/ -type f -name "*_test.py" \
	-exec realpath {} --relative-to . \;)

.PHONY: test
test:
	for t in $(TESTS); do \
		echo -e "\033[1;33m[TEST] $$t \033[0m" ; \
		python $$t || exit 1; \
		echo ; \
	done

CPU_TESTS := $(shell \
	find tests/tensorflow/data/ -type f -name "*_test.py" \
	-exec realpath {} --relative-to . \;)

.PHONY: cpu_test
cpu_test:
	for t in $(CPU_TESTS); do \
		echo -e "\033[1;33m[TEST] $$t \033[0m" ; \
		python $$t || exit 1; \
		echo ; \
	done

.PHONY: doc
doc:
	PYTHONPATH=$(shell pwd) \
	sphinx-build -M html docs/ dist/doc

.PHONY: lint
lint:
	tools/lint

.PHONY: cibuild
cibuild: lint build doc test

.PHONY: clean
clean:
	@rm -fr dist/
	@rm -fr build/
	@rm -fr *.egg-info/
	@find -name *.o -exec rm -fr {} \;
	@find -name *.d -exec rm -fr {} \;
	@find -name *.so -exec rm -fr {} \;

.DEFAULT_GOAL := build

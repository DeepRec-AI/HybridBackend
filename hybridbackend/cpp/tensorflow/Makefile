TENSORFLOW_SRC := hybridbackend/cpp/tensorflow/

TENSORFLOW_CFLAGS := \
	-DEIGEN_MPL2_ONLY \
	-DEIGEN_MAX_ALIGN_BYTES=64 \
	-DEIGEN_HAS_TYPE_TRAITS=0 \
	$(shell \
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -c \
	"import tensorflow as tf; cflags=tf.sysconfig.get_compile_flags(); print(' '.join([c.replace('-I', '-isystem ', 1) if c.startswith('-I') else c for c in cflags]))" 2>/dev/null)

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
TENSORFLOW_CFLAGS := $(TENSORFLOW_CFLAGS) -DGOOGLE_CUDA=1
endif

TENSORFLOW_LDFLAGS := \
	-Wl,-rpath='$$ORIGIN/..:$$ORIGIN/../../tensorflow' \
	$(shell \
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -c \
	"import tensorflow as tf; ldflags=tf.sysconfig.get_link_flags(); print(' '.join(ldflags))" 2>/dev/null)

TENSORFLOW_CC_SOURCES := $(shell \
	find $(TENSORFLOW_SRC) -type f \
	\( -name "*.cc" ! -name "*.cu*" \) \
	-exec realpath {} --relative-to . \;)

TENSORFLOW_OBJS := $(TENSORFLOW_CC_SOURCES:.cc=.o)
$(TENSORFLOW_OBJS): %.o:%.cc $(THIRDPARTY_DEPS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(TENSORFLOW_CFLAGS) $(CXX_CFLAGS) \
	-MMD -MP -MF $<.d -o $@ -c $< -fpic
	sed -i '/site-packages/d' $<.d
	sed -i '/^$$/N;/^\n$$/D' $<.d

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
TENSORFLOW_CU_SOURCES := $(shell \
	find $(TENSORFLOW_SRC) -type f \
	\( -name '*.cu.cc' \) \
	-exec realpath {} --relative-to . \;)

TENSORFLOW_CU_OBJS := $(TENSORFLOW_CU_SOURCES:.cc=.o)
$(TENSORFLOW_CU_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	@$(NVCC) -M $< $(CFLAGS) $(TENSORFLOW_CFLAGS) -x cu \
	 | grep -v '/usr/' \
	 | grep -v 'site-packages' \
	 | sed 's|$(notdir $@)|$@|g' \
	 | sed 's|^\./||g' \
	 > $<.d
	sed -i '/^$$/N;/^\n$$/D' $<.d
	$(NVCC) $(NVCC_CFLAGS) \
		-o $@ -c $< $(CFLAGS) $(TENSORFLOW_CFLAGS) -x cu \
		-Xcompiler -fPIC
TENSORFLOW_ALL_OBJS := $(TENSORFLOW_OBJS) $(TENSORFLOW_CU_OBJS)
else
TENSORFLOW_ALL_OBJS := $(TENSORFLOW_OBJS)
endif

$(TENSORFLOW_LIB): $(TENSORFLOW_ALL_OBJS) $(COMMON_LIB)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
	-o $@ $(TENSORFLOW_ALL_OBJS) \
	$(LDFLAGS) \
	$(TENSORFLOW_LDFLAGS) \
	-L$(LIBNAME)/ -l$(LIBNAME) \
	-lssl -lcrypto -fpic

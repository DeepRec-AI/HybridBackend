TENSORFLOW_SRC := hybridbackend/cpp/tensorflow/

ifeq ($(HYBRIDBACKEND_WITH_BUILDINFO),ON)
HYBRIDBACKEND_BUILD_FRAMEWORK := $(shell \
	$(PYTHON) -c \
	"import tensorflow as tf; print('tensorflow-%s' % tf.__git_version__)" \
	2>/dev/null)
CFLAGS := $(CFLAGS) \
	-DHYBRIDBACKEND_BUILD_FRAMEWORK="\"$(HYBRIDBACKEND_BUILD_FRAMEWORK)\""
endif

TENSORFLOW_CFLAGS := \
	-DEIGEN_MPL2_ONLY \
	-DEIGEN_MAX_ALIGN_BYTES=64 \
	-DEIGEN_HAS_TYPE_TRAITS=0 \
	$(shell \
	$(PYTHON) -c \
	"import tensorflow as tf; cflags=tf.sysconfig.get_compile_flags(); print(' '.join([c.replace('-I', '-isystem ', 1) if c.startswith('-I') else c for c in cflags]))" 2>/dev/null)

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
TENSORFLOW_CFLAGS := $(TENSORFLOW_CFLAGS) -DGOOGLE_CUDA=1
endif

ifeq ($(OS),Darwin)
TENSORFLOW_LDFLAGS := \
	$(shell \
	$(PYTHON) -c \
	"import tensorflow as tf; ldflags=tf.sysconfig.get_link_flags(); print(' '.join(ldflags))" 2>/dev/null)
TENSORFLOW_LDFLAGS := $(subst -l:libtensorflow_framework.1.dylib,-ltensorflow_framework,$(TENSORFLOW_LDFLAGS))
else
 TENSORFLOW_LDFLAGS := \
	-Wl,-rpath='$$ORIGIN/..:$$ORIGIN/../../tensorflow' \
 	$(shell \
	$(PYTHON) -c \
 	"import tensorflow as tf; ldflags=tf.sysconfig.get_link_flags(); print(' '.join(ldflags))" 2>/dev/null)
endif

TENSORFLOW_CC_SOURCES := $(shell \
	find $(TENSORFLOW_SRC) -type f \
	\( -name "*.cc" ! -name "*.cu*" \) \
	-exec realpath {} --relative-to . \;)

TENSORFLOW_OBJS := $(TENSORFLOW_CC_SOURCES:.cc=.o)
ifeq ($(OS),Darwin)
$(TENSORFLOW_OBJS): %.o:%.cc $(THIRDPARTY_DEPS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(TENSORFLOW_CFLAGS) $(CXX_CFLAGS) \
	-MMD -MP -MF $<.d -o $@ -c $< -fpic
else
$(TENSORFLOW_OBJS): %.o:%.cc $(THIRDPARTY_DEPS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(TENSORFLOW_CFLAGS) $(CXX_CFLAGS) \
	-MMD -MP -MF $<.d -o $@ -c $< -fpic
	sed -i '/site-packages/d' $<.d
	sed -i '/^$$/N;/^\n$$/D' $<.d
endif

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
TENSORFLOW_CU_SOURCES := $(shell \
	find $(TENSORFLOW_SRC) -type f \
	\( -name '*.cu.cc' \) \
	-exec realpath {} --relative-to . \;)

TENSORFLOW_CU_OBJS := $(TENSORFLOW_CU_SOURCES:.cc=.o)
ifeq ($(OS),Darwin)
$(TENSORFLOW_CU_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) \
		-o $@ -c $< $(CFLAGS) $(TENSORFLOW_CFLAGS) -x cu \
		-Xcompiler -fPIC
else
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
endif
TENSORFLOW_ALL_OBJS := $(TENSORFLOW_OBJS) $(TENSORFLOW_CU_OBJS)
else
TENSORFLOW_ALL_OBJS := $(TENSORFLOW_OBJS)
endif

ifeq ($(OS),Darwin)
$(TENSORFLOW_LIB): $(TENSORFLOW_ALL_OBJS) $(COMMON_LIB)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
	-install_name @rpath/lib$(LIBNAME)_tensorflow.so \
	-o $@ $(TENSORFLOW_ALL_OBJS) \
	$(LDFLAGS) \
	$(TENSORFLOW_LDFLAGS) \
	-L$(LIBNAME)/ -l$(LIBNAME)
else
$(TENSORFLOW_LIB): $(TENSORFLOW_ALL_OBJS) $(COMMON_LIB)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
	-o $@ $(TENSORFLOW_ALL_OBJS) \
	$(LDFLAGS) \
	$(TENSORFLOW_LDFLAGS) \
	-L$(LIBNAME)/ -l$(LIBNAME)
endif

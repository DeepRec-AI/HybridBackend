COMMON_SRC := hybridbackend/common/

COMMON_CFLAGS := \
	$(shell $(PYTHON) -m pybind11 --includes)

COMMON_CC_SOURCES := $(shell \
	find $(COMMON_SRC) -type f \
	\( -name "*.cc" ! -name "*.cu*" \) \
	-exec realpath {} --relative-to . \;)

COMMON_OBJS := $(COMMON_CC_SOURCES:.cc=.o)
$(COMMON_OBJS): %.o:%.cc $(THIRDPARTY_DEPS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(CXX_CFLAGS) $(COMMON_CFLAGS) \
	-MMD -MP -MF $<.d -o $@ -c $< -fpic

ifeq ($(HYBRIDBACKEND_WITH_CUDA),ON)
COMMON_CU_SOURCES := $(shell \
	find $(COMMON_SRC) -type f \
	\( -name '*.cu.cc' \) \
	-exec realpath {} --relative-to . \;)

COMMON_CU_OBJS := $(COMMON_CU_SOURCES:.cc=.o)
ifeq ($(OS),Darwin)
$(COMMON_CU_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_CFLAGS) \
		-o $@ -c $< $(CFLAGS) $(COMMON_CFLAGS) -x cu \
		-Xcompiler -fPIC
else
$(COMMON_CU_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	@$(NVCC) -M $< $(CFLAGS) $(COMMON_CFLAGS) -x cu \
	 | grep -v '/usr/' \
	 | sed 's|$(notdir $@)|$@|g' \
	 | sed 's|\./||g' \
	 > $<.d
	$(NVCC) $(NVCC_CFLAGS) \
		-o $@ -c $< $(CFLAGS) $(COMMON_CFLAGS) -x cu \
		-Xcompiler -fPIC
endif
COMMON_ALL_OBJS := $(COMMON_OBJS) $(COMMON_CU_OBJS)
else
COMMON_ALL_OBJS := $(COMMON_OBJS)
endif

ifeq ($(OS),Darwin)
$(COMMON_LIB): $(COMMON_ALL_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(COMMON_CFLAGS) -std=c++11 \
	-install_name @rpath/lib$(LIBNAME).so \
	-framework CoreFoundation \
	-o $@ $^ $(LDFLAGS) $(COMMON_LDFLAGS)
else
$(COMMON_LIB): $(COMMON_ALL_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(COMMON_CFLAGS) -std=c++11 \
	-o $@ $^ $(LDFLAGS) $(COMMON_LDFLAGS)
endif

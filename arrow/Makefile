ARROW_BUILDDIR := $(shell pwd)/arrow/build
ARROW_DISTDIR := $(shell pwd)/arrow/dist
ARROW_CMAKE_ARGS := ""

$(ARROW_BUILDDIR)/install_manifest.txt: \
	arrow/build.sh
	mkdir -p $(dir $@)
	ARROW_INSTALL=$(ARROW_DISTDIR) \
	ARROW_BUILD=$(ARROW_BUILDDIR) \
	ARROW_OSX_TARGET=$(OSX_TARGET) \
	USE_CXX11_ABI=$(HYBRIDBACKEND_USE_CXX11_ABI) \
	WITH_ARROW_HDFS=$(HYBRIDBACKEND_WITH_ARROW_HDFS) \
	WITH_ARROW_S3=$(HYBRIDBACKEND_WITH_ARROW_S3) \
	SIMD_LEVEL=$(HYBRIDBACKEND_WITH_ARROW_SIMD_LEVEL) \
	OS=$(OS) \
	bash arrow/build.sh

ARROW_LIB := $(ARROW_BUILDDIR)/include/arrow/api.h
$(ARROW_LIB): \
	$(ARROW_BUILDDIR)/install_manifest.txt
	cd $(ARROW_BUILDDIR) && \
	$(MAKE) install

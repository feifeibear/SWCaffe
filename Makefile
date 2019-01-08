####COMPILERS####
LINK 	= 	mpiCC
CXX 	=  	mpiCC -host -CG:pjump_all
SWHCXX = sw5cc.new -host -msimd
SWSCXX = 	sw5cc.new -slave -CG:pjump_all -msimd

####FLAGS####
#basic compile flags
FLAGS = 	-O2 -OPT:IEEE_arith=2 -OPT:Olimit=0
#caffe compile flags
FLAGS += 	-DCPU_ONLY
FLAGS +=  -DUSE_OPENCV
FLAGS +=  -DUSE_LMDB
#swcaffe compile flags
#FLAGS += 	-DSWMPI
#support prefetch data for SWMPI
#FLAGS +=  -DDATAPREFETCH
#FLAGS +=  -DSWMPITEST
#4CG Support
#only SW4CG mode: 1CG on 4CGs
#FLAGS += -DSW4CG
#4CG support for each layer
#FLAGS += -DSW4CG_CONV_FW
#FLAGS += -DSW4CG_CONV_BW
#swdnn flags
FLAGS += -DUSE_SWDNN
FLAGS += -DSW_TRANS
#use sw based basic functions in src/caffe/util/math_functions.cpp
FLAGS += -DUSE_SWBASE
FLAGS += -DUSE_SWPOOL
FLAGS += -DUSE_SWRELU
FLAGS += -DUSE_SWIM2COL
FLAGS += -DUSE_SWPRELU
FLAGS += -DUSE_SWSOFTMAX
FLAGS += -DUSE_SWBATCHNORM
FLAGS += -DUSE_BIAS

#FLAGS += -DDEBUG_SWBASE
#FLAGS += -DDEBUG_PRINT_TIME

#debug flags
#alogrithm logic and forbackward time
#FLAGS += 	-DDEBUG_VERBOSE_1
#time of each layer
FLAGS += 	-DDEBUG_VERBOSE_2
#print timer in sw_conv_layer_impl
#FLAGS += 	-DDEBUG_VERBOSE_3
#address and length of mpibuff
#FLAGS += 	-DDEBUG_VERBOSE_6
#in sgd solvers data value print
#FLAGS +=  -DDEBUG_VERBOSE_7
#debug SW4CG
#FLAGS += -DDEBUG_VERBOSE_8
#FLAGS += -DDEBUG_SYNC_4CG
#debug SWDNN
#FLAGS += -DDEBUG_VERBOSE_9
#FLAGS += -DDEBUG_VERBOSE_SWDNN

#basic load flags with sw5cc
LDFLAGS = -lm_slave 
# uncomment for 1CG
LDFLAGS += -allshare
#include flags
SWINC_FLAGS=-I./include -I$(THIRD_PARTY_DIR)/include
SWINC_FLAGS+=-I./swdnn_release/include

####DIRS####
SRC = ./src
SWBUILD_DIR=./swbuild
THIRD_PARTY_DIR=../thirdparty
BIN_DIR=./bin

####SRC####
caffesrc=$(wildcard ./src/caffe/*.cpp ./src/caffe/layers/*.cpp ./src/caffe/solvers/*.cpp ./src/caffe/util/*.cpp ./src/caffe/ringTreeAllreduce/*.cpp ./src/glog/*.cpp) 
caffepbsrc = ./src/caffe/proto/caffe.pb.cc
swhostsrc = $(wildcard ./src/caffe/swutil/*.c ./src/caffe/ringTreeAllreduce/*.c)
swslavesrc = $(wildcard ./src/caffe/swutil/slave/*.c ./src/caffe/ringTreeAllreduce/slave/*.c)
swslavessrc = $(wildcard ./src/caffe/swutil/slave/*.S)

####OBJS####
caffeobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.cpp, %.o, $(caffesrc)))
caffepbobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.cc, %.o, $(caffepbsrc)))
swhostobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.c, %.o, $(swhostsrc)))
swslaveobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.c, %.o, $(swslavesrc)))
swslavesobjs = $(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.S, %_asm.o, $(swslavessrc)))
allobjs = $(caffeobjs) $(caffepbobjs) $(swhostobjs) $(swslaveobjs) $(swslavesobjs)

#libraries
SWLIBOBJ=$(THIRD_PARTY_DIR)/lib/cblas_LINUX0324.a
#swblas for 4CG(all share mode!!)
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libswblasall-2.a
#swblas for 1cg
#SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libswblas0324.a


SWLIBOBJ+=-Wl,--whole-archive $(THIRD_PARTY_DIR)/lib/libopencv_core.a -Wl,--no-whole-archive
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libopencv_highgui.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libopencv_imgproc.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libjpeg.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libz.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libprotobuf.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libboost_system.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libboost_thread.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libboost_atomic.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libgflags.a
#######order matters
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/liblmdb.a
#SWLIBOBJ+=/home/export/online1/swyf/swdnn/fjr/2018-09/BLAS/swDNNv2.0/build/libswdnnlib.a
SWLIBOBJ+=./swdnn_release/libswdnnlib.a

####Rules####
#debug makefile
show:
	echo $(caffesrc)
	echo $(caffepbsrc)
	echo $(swhostsrc)
	echo $(swslavesrc)
	echo $(swslavessrc)
	echo $(caffeobjs)
	echo $(caffepbobjs)
	echo $(swhostobjs)
	echo $(swslaveobjs)
	echo $(swslavesobjs)

caffe: $(BIN_DIR)/caffe_sw
lstm: $(BIN_DIR)/lstm_sequence
convert_imageset: $(BIN_DIR)/convert_imageset_sw
ar: $(allobjs)
	swar rcs ./lib/sw/swcaffe.a $^
mk:
	mkdir -p $(SWBUILD_DIR) $(SWBUILD_DIR)/caffe $(SWBUILD_DIR)/caffe/util $(SWBUILD_DIR)/caffe/layers \
		$(SWBUILD_DIR)/caffe/swutil $(SWBUILD_DIR)/caffe/swutil/slave $(SWBUILD_DIR)/caffe/proto\
		$(SWBUILD_DIR)/caffe/solvers $(SWBUILD_DIR)/glog  $(BIN_DIR) ./lib/sw\
		$(SWBUILD_DIR)/caffe/ringTreeAllreduce $(SWBUILD_DIR)/caffe/ringTreeAllreduce/slave
#caffe tools
$(BIN_DIR)/convert_imageset_sw: $(SWBUILD_DIR)/swobj/convert_imageset_sw.o $(allobjs)
	$(LINK) $^ $(LDFLAGS)  -o $@ $(SWLIBOBJ)
$(SWBUILD_DIR)/convert_imageset_sw.o: ./tools/convert_imageset.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/caffe_sw:$(allobjs) $(SWBUILD_DIR)/caffe_sw.o
	$(LINK) $^ $(LDFLAGS) -o $@ $(SWLIBOBJ)
$(SWBUILD_DIR)/caffe_sw.o: ./tools/caffe.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/lstm_sequence: $(allobjs) $(SWBUILD_DIR)/lstm_sequence.o
	$(LINK) $^ $(LDFLAGS) -o $@ $(SWLIBOBJ)
$(SWBUILD_DIR)/lstm_sequence.o: ./tools/lstm_sequence.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(SWBUILD_DIR)/caffe/swutil/slave/gemm_asm.o: ./src/caffe/swutil/slave/gemm.S
	$(SWSCXX) $(FLAGS) $(SWINC_FLAGS) -c $< -o $@
$(SWBUILD_DIR)/caffe/swutil/slave/gemm_float_asm.o: ./src/caffe/swutil/slave/gemm_float.S
	$(SWSCXX) $(FLAGS) $(SWINC_FLAGS) -c $< -o $@
$(SWBUILD_DIR)/caffe/swutil/slave/%.o: ./src/caffe/swutil/slave/%.c
	$(SWSCXX) -c $< $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/swutil/%.o: ./src/caffe/swutil/%.c
	$(SWHCXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/ringTreeAllreduce/%.o: ./src/caffe/ringTreeAllreduce/%.c
	sw5cc.new -host -c -msimd -O2 $(SWINC_FLAGS) $^ -o $@
$(SWBUILD_DIR)/caffe/ringTreeAllreduce/slave/sw_slave_add.o: ./src/caffe/ringTreeAllreduce/slave/sw_slave_add.c
	sw5cc.new -slave -c -msimd -O2 ./src/caffe/ringTreeAllreduce/slave/sw_slave_add.c -o $(SWBUILD_DIR)/caffe/ringTreeAllreduce/slave/sw_slave_add.o

$(SWBUILD_DIR)/caffe/%.o: ./src/caffe/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/layers/%.o: ./src/caffe/layers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/solvers/%.o: ./src/caffe/solvers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/%.o: ./src/caffe/util/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/glog/%.o: ./src/glog/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/proto/%.o: ./src/caffe/proto/%.cc
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/ringTreeAllreduce/%.o: ./src/caffe/ringTreeAllreduce/%.cpp
	$(CXX) -c -host -O2 -DSWMPI -I. $(SWINC_FLAGS) $^ -o $@

clean:
	rm -f $(allobjs) $(SWBUILD_DIR)/*.o core* $(BIN_DIR)/*

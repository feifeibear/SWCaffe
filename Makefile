CXX 	=  	mpiCC -host -CG:pjump_all 
LINK 	= 	mpiCC 
SWCXX = 	sw5cc.new -slave -CG:pjump_all
FLAGS = 	-O2 -OPT:IEEE_arith=2 -OPT:Olimit=0 
FLAGS += 	-DCPU_ONLY
FLAGS += 	-DSWMPI
LDFLAGS = -lm_slave 
LDFLAGS += -allshare

#alogrithm logic and forbackward time
FLAGS += 	-DDEBUG_VERBOSE_1
#time of each layer
#FLAGS += 	-DDEBUG_VERBOSE_2
#print timer in sw_conv_layer_impl
#FLAGS += 	-DDEBUG_VERBOSE_3
#address and length of mpibuff
#FLAGS += 	-DDEBUG_VERBOSE_6
#in sgd solvers data value print
#FLAGS +=  -DDEBUG_VERBOSE_7


SWBUILD_DIR=./swbuild
THIRD_PARTY_DIR=../thirdparty
SWINC_FLAGS=-I./include -I$(THIRD_PARTY_DIR)/include

SWLIBOBJ=$(THIRD_PARTY_DIR)/lib/cblas_LINUX0324.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libswblas0324.a
#SWLIBOBJ+=-Wl,--whole-archive $(THIRD_PARTY_DIR)/lib/libhdf5.a
#SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/libhdf5_hl.a -Wl,--no-whole-archive
SWLIBOBJ+=-Wl,--whole-archive $(THIRD_PARTY_DIR)/lib/libopencv_core.a -Wl,--no-whole-archive
FLAGS += -DUSE_OPENCV
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
FLAGS += -DUSE_LMDB
SWLIBOBJ+=$(THIRD_PARTY_DIR)/lib/liblmdb.a

src=$(wildcard ./src/caffe/*.cpp ./src/caffe/layers/*.cpp ./src/caffe/solvers/*.cpp ./src/caffe/util/*.cpp ./src/glog/*.cpp)
SWOBJ=$(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.cpp, %.o, $(src))) $(SWBUILD_DIR)/caffe/proto/caffe.pb.o
swdnnsrc=$(wildcard ./src/caffe/swlayers/*.c ./src/caffe/util/*.c)
SWDNNOBJ=$(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.c, %.o, $(swdnnsrc)))
SWDNNOBJ+=$(SWBUILD_DIR)/caffe/swlayers/gemm_asm.o
SWDNNOBJ+=$(SWBUILD_DIR)/caffe/swlayers/gemm_asm_float.o
SWOBJ+=$(SWDNNOBJ)
#FLAGS += -DUSE_SWDNN
FLAGS += -DSW_TRANS
FLAGS += -DUSE_SWPOOL
FLAGS += -DUSE_SWRELU
FLAGS += -DUSE_SWIM2COL
FLAGS += -DUSE_SWPRELU
FLAGS += -DUSE_SWSOFTMAX
FLAGS += -DDEBUG_PRINT_TIME
BIN_DIR=./bin
video: $(BIN_DIR)/video_sw
caffe: $(BIN_DIR)/caffe_sw
lenet: $(BIN_DIR)/test_lenet_sw
vgg: $(BIN_DIR)/vggnet_sw
alexnet: $(BIN_DIR)/alexnet_sw
solver: $(BIN_DIR)/test_solver_sw
lstm: $(BIN_DIR)/test_lstm_sw
compute_image_mean: $(BIN_DIR)/compute_image_mean_sw
convert_imageset: $(BIN_DIR)/convert_imageset_sw

mk:
	mkdir -p $(SWBUILD_DIR) $(SWBUILD_DIR)/caffe $(SWBUILD_DIR)/caffe/util $(SWBUILD_DIR)/caffe/layers $(SWBUILD_DIR)/caffe/swlayers $(SWBUILD_DIR)/caffe/proto\
		$(SWBUILD_DIR)/caffe/solvers $(SWBUILD_DIR)/glog ./models/swobj $(BIN_DIR)
	mkdir -p lib/sw
ar: $(SWOBJ) $(SWLIBOBJ)
	swar rcs ./lib/sw/swcaffe.a $^  

runvideo:
	sh ./scripts/sw_runvideo.sh 1
runalex:
	sh ./scripts/sw_runalex.sh 1
runvgg:
	sh ./scripts/sw_runvgg.sh 1
runlenet:
	sh ./scripts/sw_runlenet.sh 1
runsolver:
	sh ./scripts/sw_runsolver.sh 1
runlstm:
	sh ./scripts/sw_runlstm.sh 1

$(BIN_DIR)/convert_imageset_sw: ./models/swobj/convert_imageset_sw.o $(SWOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@ $(SWLIBOBJ)
./models/swobj/convert_imageset_sw.o: ./tools/convert_imageset.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@


$(BIN_DIR)/caffe_sw: ./models/swobj/caffe_sw.o $(SWOBJ) 
	$(LINK) $^ $(LDFLAGS)  -o $@ $(SWLIBOBJ)
./models/swobj/caffe_sw.o: ./tools/caffe.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/video_sw: ./models/swobj/videoprocess.o $(SWOBJ) 
	$(LINK) $^ $(LDFLAGS)  -o $@ $(SWLIBOBJ)
./models/swobj/videoprocess.o: ./models/src/videoprocess.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/alexnet_sw: ./models/swobj/alexnet.o $(SWOBJ) 
	$(LINK) $^ $(LDFLAGS)  -o $@ $(SWLIBOBJ)
./models/swobj/alexnet.o: ./models/src/alexnet.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/vggnet_sw: ./models/swobj/vggnet.o $(SWOBJ) 
	$(LINK) $^   -o $@ $(LDFLAGS) $(SWLIBOBJ)
./models/swobj/vggnet.o: ./models/src/vggnet.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/test_solver_sw: ./models/swobj/test_solver.o $(SWOBJ) $(SWLIBOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@
./models/swobj/test_solver.o: ./models/src/test_solver.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/test_lenet_sw: ./models/swobj/test_lenet.o $(SWOBJ) $(SWLIBOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@
./models/swobj/test_lenet.o: ./models/src/test_lenet.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(BIN_DIR)/test_lstm_sw: ./models/swobj/test_lstm.o $(SWOBJ) $(SWLIBOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@
./models/swobj/test_lstm.o: ./models/src/test_lstm.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(SWBUILD_DIR)/caffe/swlayers/sw_pool_layer_impl.o: ./src/caffe/swlayers/sw_pool_layer_impl.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_pool.o: ./src/caffe/swlayers/sw_slave_pool.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@

$(SWBUILD_DIR)/caffe/swlayers/sw_slave_pool_f.o: ./src/caffe/swlayers/sw_slave_pool_f.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@

$(SWBUILD_DIR)/caffe/util/swmatrix_trans.o: ./src/caffe/util/swmatrix_trans.c
	sw5cc.new -slave -msimd -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/swmatrix_trans_f.o: ./src/caffe/util/swmatrix_trans_f.c
	sw5cc.new -slave -msimd -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(SWBUILD_DIR)/caffe/util/sw_slave_im2col.o: ./src/caffe/util/sw_slave_im2col.c
	sw5cc.new -slave -msimd -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/swim2col.o: ./src/caffe/util/swim2col.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(SWBUILD_DIR)/caffe/util/sw_slave_imgnet_data_copy.o: ./src/caffe/util/sw_slave_imgnet_data_copy.c
	sw5cc.new -slave -msimd -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/swimgnet_data_copy.o: ./src/caffe/util/swimgnet_data_copy.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(SWBUILD_DIR)/caffe/util/matrix_trans.o: ./src/caffe/util/matrix_trans.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/swdata_type_trans.o: ./src/caffe/util/swdata_type_trans.c
	sw5cc.new -slave -msimd -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/data_type_trans.o: ./src/caffe/util/data_type_trans.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/sw_slave_memcpy.o: ./src/caffe/util/sw_slave_memcpy.c
	sw5cc.new -slave -msimd -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/sw_memcpy.o: ./src/caffe/util/sw_memcpy.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_conv_layer_impl.o: ./src/caffe/swlayers/sw_conv_layer_impl.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_conv_valid.o: ./src/caffe/swlayers/sw_slave_conv_valid.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_conv_full.o: ./src/caffe/swlayers/sw_slave_conv_full.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_conv_pad_float.o: ./src/caffe/swlayers/sw_slave_conv_pad_float.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_conv_full_pad_float_v2.o: ./src/caffe/swlayers/sw_slave_conv_full_pad_float_v2.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_conv_pad_float2double.o: ./src/caffe/swlayers/sw_slave_conv_pad_float2double.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_conv_pad.o: ./src/caffe/swlayers/sw_slave_conv_pad.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_conv_full_pad_float.o: ./src/caffe/swlayers/sw_slave_conv_full_pad_float.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_conv_full_pad.o: ./src/caffe/swlayers/sw_slave_conv_full_pad.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/gemm_asm.o: ./src/caffe/swlayers/gemm.S
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/gemm_asm_float.o: ./src/caffe/swlayers/gemm_float.S
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_relu_layer_impl.o: ./src/caffe/swlayers/sw_relu_layer_impl.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_relu.o: ./src/caffe/swlayers/sw_slave_relu.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_prelu_layer_impl.o: ./src/caffe/swlayers/sw_prelu_layer_impl.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_prelu.o: ./src/caffe/swlayers/sw_slave_prelu.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_softmax_layer_impl.o: ./src/caffe/swlayers/sw_softmax_layer_impl.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/swlayers/sw_slave_softmax.o: ./src/caffe/swlayers/sw_slave_softmax.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
# Your layers compile here

$(SWBUILD_DIR)/%.o: ./src/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/layers/%.o: ./src/caffe/layers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/%.o: ./src/caffe/util/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/util/%.o: ./src/caffe/util/%.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/solvers/%.o: ./src/caffe/solvers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/glog/%.o: ./src/glog/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/caffe/proto/%.o: ./src/caffe/proto/%.cc
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@


clean:
	rm -f $(SWOBJ) vggnet test_solver ./models/swobj/* core*

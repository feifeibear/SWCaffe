FLAGS=-O2 -DCPU_ONLY
BOOST_DIR=/home/export/online1/swyf/swdnn/softs/install

INC_FLAGS +=-I../../thirdparty/glog_install/include
INC_FLAGS += -I../../thirdparty/openblas_install/include
INC_FLAGS += -I../../thirdparty/googletest/include
INC_FLAGS += -I$(BOOST_DIR)
INC_FLAGS += -I../include -I./include

LDFLAGS += -L../../thirdparty/glog_install/lib/ -lglog
LDFLAGS += -L../../thirdparty/openblas_install/lib -lopenblas

THIRD_PARTY_DIR=../../thirdparty
SWINC_FLAGS=-I../include -I./include -I$(THIRD_PARTY_DIR)/CBLAS/include -I${BOOST_DIR}


CAFFE_OBJ = 	../swdnn_test/obj/test_caffe_main.o\
							../swdnn_test/obj/test_convolution_layer.o\
							../lib/swcaffe.a

SWDNN_OBJ= 		../swdnn_test/obj/test_swdnn_main.o\
							../swbuild/swlayers/sw_conv_layer_impl.o\
							../swbuild/swlayers/sw_slave_conv_valid.o\
							../swbuild/swlayers/sw_slave_conv_full.o\
							../swbuild/swlayers/sw_slave_conv_pad.o\
							../swbuild/swlayers/sw_slave_conv_pad_float.o\
							../swbuild/swlayers/sw_slave_conv_full_pad.o\
							../swbuild/swlayers/sw_slave_conv_full_pad_float.o\
							../swbuild/swlayers/gemm_asm.o\
							../swbuild/swlayers/gemm_asm_float.o\
							../swbuild/util/matrix_trans.o\
							../swbuild/util/swmatrix_trans.o\
							../swbuild/glog/logging.o


#../swdnn_test/obj/conv_layer_impl.o\


all: ./bin/swdnn_main
run: ./bin/swdnn_main
	bsub -b -I -m 1 -p -q q_sw_share -host_stack 1000 -share_size 5000 -n 1 -cgsp 64 ./bin/swdnn_main
./bin/caffe_main: $(CAFFE_OBJ)
	mpiicc $^ ../../thirdparty/googletest/libgtest.a -o $@ $(LDFLAGS)
./bin/swdnn_main: $(SWDNN_OBJ)
	mpiCC $^ -o $@

#intel
../swdnn_test/obj/test_caffe_main.o: ../swdnn_test/src/test_caffe_main.cpp 
	mpiicc $(FLAGS) $(INC_FLAGS) -c $^ -o $@ 
../swdnn_test/obj/test_convolution_layer.o: ../swdnn_test/src/test_convolution_layer.cpp
	mpiicc $(FLAGS) $(INC_FLAGS) -c $^ -o $@

#SW26010
../swdnn_test/obj/test_swdnn_main.o: ../swdnn_test/src/test_swdnn_main.cpp 
	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@ 
#../swdnn_test/obj/conv_layer_impl.o: ../swdnn_test/src/conv_layer_impl.cpp
#	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@

clean:
	rm ./obj/*

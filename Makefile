CXX 	=  	mpiCC -host
LINK 	= mpiCC
SWCXX = 	sw5cc.new -slave
FLAGS = 	-O2 -OPT:IEEE_arith=2
#FLAGS +=  -OPT:Olimit=6020
FLAGS += 	-DCPU_ONLY
FLAGS +=  -DMYMPI
#-DSW_CODE 

THIRD_PARTY_DIR=../thirdparty
BOOST_DIR=/home/export/online1/swyf/swdnn/softs/install
SWINC_FLAGS=-I./include -I$(THIRD_PARTY_DIR)/CBLAS/include -I${BOOST_DIR}

SWLIBOBJ=$(THIRD_PARTY_DIR)/swblas/SWCBLAS/lib/cblas_LINUX0324.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/swblas/SWCBLAS/libswblas0324.a

SWOBJ=./build/blob.o ./build/common.o ./build/syncedmem.o ./build/layer_factory.o\
		./build/util/math_functions.o \
		./build/util/insert_splits.o \
		./build/util/im2col.o \
		./build/layers/inner_product_layer.o \
		./build/layers/input_layer.o \
		./build/layers/base_conv_layer.o \
		./build/layers/conv_layer.o \
		./build/layers/pooling_layer.o \
		./build/layers/data_layer.o \
		./build/layers/imagenet_data_layer.o \
		./build/layers/mnist_data_layer.o \
		./build/layers/neuron_layer.o\
		./build/layers/relu_layer.o\
		./build/layers/softmax_layer.o\
		./build/layers/softmax_loss_layer.o\
		./build/layers/loss_layer.o\
		./build/layers/accuracy_layer.o\
		./build/layers/split_layer.o\
		./build/layers/default_instance.o\
		./build/layers/lstm_layer.o\
		./build/layers/lstm_unit_layer.o\
		./build/layers/recurrent_layer.o\
		./build/layers/eltwise_layer.o\
		./build/layers/scale_layer.o\
		./build/layers/slice_layer.o\
		./build/layers/concat_layer.o\
		./build/layers/reshape_layer.o\
		./build/layers/bias_layer.o\
		./build/layers/reduction_layer.o\
		./build/layers/euclidean_loss_layer.o\
		./build/layers/silence_layer.o\
		./build/layers/dropout_layer.o\
		./build/net.o\
		./build/solvers/rmsprop_solver.o\
		./build/solvers/sgd_solver.o\
		./build/util/benchmark.o\
		./build/solver.o\
		./build/util/mpi.o\
	  ./build/swlayers/sw_slave_conv_valid.o\
	  ./build/swlayers/sw_slave_conv_full.o\
	  ./build/swlayers/gemm_asm.o\
		./build/swlayers/sw_conv_layer_impl.o\
		./build/glog/logging.o\
		./build/util/matrix_trans.o\
		./build/util/swmatrix_trans.o

all: mk test_solver 

mk:
	mkdir -p build build/util build/layers build/swlayers build/solvers build/glog 

run:
	bsub -b -I -m 1 -p -q q_sw_share -host_stack 2048 -share_size 5000 -n 1 -cgsp 64 ./vggnet
runlenet:
	bsub -b -I -m 1 -p -q q_sw_share -host_stack 1024 -share_size 6000 -n 1 -cgsp 64 ./test_solver

vggnet: ./models/obj/vggnet.o $(SWOBJ) $(SWLIBOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@
./models/obj/vggnet.o: ./models/src/vggnet.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

test_solver: ./models/obj/test_solver.o $(SWOBJ) $(SWLIBOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@
./models/obj/test_solver.o: ./models/src/test_solver.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

./build/util/swmatrix_trans.o: ./src/util/swmaxtrix_trans.c
	sw5cc.new -slave -msimd $(FLAGS) $(SWINC_FLAGS) -c $< -o $@
./build/util/matrix_trans.o: ./src/util/matrix_trans.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
./build/swlayers/sw_conv_layer_impl.o: ./src/swlayers/sw_conv_layer_impl.c
	sw5cc.new -host $(FLAGS) $(SWINC_FLAGS) -c $< -o $@
./build/swlayers/sw_slave_conv_valid.o: ./src/swlayers/sw_slave_conv_valid.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
./build/swlayers/sw_slave_conv_full.o: ./src/swlayers/sw_slave_conv_full.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
./build/swlayers/gemm_asm.o: ./src/swlayers/gemm.S
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@

./build/%.o: ./src/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
./build/layers/%.o: ./src/layers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
./build/util/%.o: ./src/util/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
./build/solvers/%.o: ./src/solvers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
./build/glog/%.o: ./src/glog/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

clean:
	rm -f $(SWOBJ) vggnet test_solver ./models/obj/* core*

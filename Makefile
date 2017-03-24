CXX = mpiCC -host
LINK = mpiCC
FLAGS=-O3
FLAGS+=-DCPU_ONLY

INC_FLAGS = -I ./include
INC_FLAGS += -I ../../tools/CBLAS/include
#INC_FLAGS += -I ../../local/swhdf5/include


LIBOBJ = ../../tools/swblas/SWCBLAS/lib/cblas_LINUX0324.a
LIBOBJ += ../../tools/swblas/SWCBLAS/libswblas0324.a
#LIBOBJ +=	../../local/swhdf5/lib/libhdf5.a
#-lhdf5_hl

OBJ=./build/blob.o ./build/common.o ./build/syncedmem.o ./build/layer_factory.o\
		./build/util/math_functions.o \
		./build/util/insert_splits.o \
		./build/util/im2col.o \
		./build/layers/inner_product_layer.o \
		./build/layers/input_layer.o \
		./build/layers/base_conv_layer.o \
		./build/layers/conv_layer.o \
		./build/layers/pooling_layer.o \
		./build/layers/data_layer.o \
		./build/layers/neuron_layer.o\
		./build/layers/relu_layer.o\
		./build/layers/softmax_layer.o\
		./build/layers/softmax_loss_layer.o\
		./build/layers/loss_layer.o\
		./build/layers/accuracy_layer.o\
		./build/layers/split_layer.o\
		./build/layers/default_instance.o\
		./build/net.o\
		./build/solvers/sgd_solver.o\
		./build/util/benchmark.o\
		./build/solver.o\
		./build/glog/logging.o
		#./build/solvers/adadelta_solver.o\
		#./build/solvers/adagrad_solver.o\
		#./build/solvers/adam_solver.o\
		#./build/solvers/nesterov_solver.o\
		#./build/solvers/rmsprop_solver.o\
		#./build/util/hdf5.o \

TEST_OBJ=./build/blob.o ./build/common.o ./build/syncedmem.o ./build/layer_factory.o\
		./build/util/math_functions.o \
		./build/util/insert_splits.o \
		./build/util/im2col.o \
		./build/util/hdf5.o \
		./build/layers/inner_product_layer.o \
		./build/layers/input_layer.o \
		./build/layers/base_conv_layer.o \
		./build/layers/conv_layer.o \
		./build/layers/pooling_layer.o \
		./build/layers/data_layer.o \
		./build/layers/neuron_layer.o\
		./build/layers/relu_layer.o\
		./build/layers/softmax_layer.o\
		./build/layers/softmax_loss_layer.o\
		./build/layers/loss_layer.o\
		./build/layers/accuracy_layer.o\
		./build/layers/split_layer.o\
		./build/layers/default_instance.o\
		./build/net.o\
		./build/solvers/adadelta_solver.o\
		./build/solvers/adagrad_solver.o\
		./build/solvers/adam_solver.o\
		./build/solvers/nesterov_solver.o\
		./build/solvers/rmsprop_solver.o\
		./build/solvers/sgd_solver.o\
		./build/util/benchmark.o\
		./build/solver.o\
		./build/test/test_caffe_main.o\
		./build/test/test_accuracy_layer.o\
		./build/test/test_blob.o\
		./build/test/test_common.o\
		./build/test/test_convolution_layer.o\
		./build/test/test_filler.o\
		./build/test/test_inner_product_layer.o\
		./build/test/test_math_functions.o\
		./build/test/test_pooling_layer.o\
		./build/test/test_softmax_layer.o\
		./build/test/test_softmax_with_loss_layer.o\
		./build/test/test_syncedmem.o

all:test_solver
run: test_solver
	bsub -b -I -m 1 -p -q q_sw_share -host_stack 1024 -share_size 6000 -n 1 -cgsp 64 ./$^

test_solver: test_solver.o $(OBJ) $(LIBOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@
#-lhdf5_cpp -lhdf5_hl_cpp  
test_solver.o: test_solver.cpp
	$(CXX) -c $^ $(FLAGS) $(INC_FLAGS) -o $@
#
#test_all: $(OBJ)
#	$(CXX) -pthread $^ ../thirdparty/googletest/libgtest.a \
#	-L ../thirdparty/glog_install/lib/ -L ../thirdparty/openblas_install/lib \
#	-L ../thirdparty/hdf5_install/lib -lglog -lopenblas -lhdf5 -lhdf5_cpp -lhdf5_hl -lhdf5_hl_cpp -o $@

./build/%.o: ./src/%.cpp
	$(CXX) -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/layers/%.o: ./src/layers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/util/%.o: ./src/util/%.cpp
	$(CXX) -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/solvers/%.o: ./src/solvers/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/glog/%.o: ./src/glog/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@

./build/test/%.o: ./src/test/%.cpp
	$(CXX) -c $^ $(FLAGS) $(INC_FLAGS) -isystem ../thirdparty/googletest/include/ -o $@

clean:
	rm *.o ./build/*.o ./build/layers/*.o ./build/util/*.o ./build/solvers/*.o ./build/test/*.o test testcp test_solver test_all

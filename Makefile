FLAGS=-O3
FLAGS+=-DCPU_ONLY

INC_FLAGS=-I../thirdparty/glog_install/include
INC_FLAGS += -I../thirdparty/openblas_install/include
INC_FLAGS += -I../thirdparty/hdf5_install/include
INC_FLAGS += -I../thirdparty/boost_install/include
INC_FLAGS += -I./include

TEST_INC_FLAGS=-I../thirdparty/glog_install/include
TEST_INC_FLAGS += -I../thirdparty/openblas_install/include
TEST_INC_FLAGS += -I../thirdparty/hdf5_install/include
TEST_INC_FLAGS += -I./include
TEST_INC_FLAGS += -I../thirdparty/googletest/include

LDFLAGS += -L ../thirdparty/glog_install/lib/ -lglog
LDFLAGS += -L ../thirdparty/openblas_install/lib -lopenblas
LDFLAGS += -L ../thirdparty/hdf5_install/lib -lhdf5 -lhdf5_hl
LDFLAGS += -L ../thirdparty/boost_install/lib -lboost_system
LDFLAGS += -L ../thirdparty/boost_install/lib -lboost_serialization

OBJ=./build/blob.o ./build/common.o ./build/syncedmem.o ./build/layer_factory.o\
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
		./build/solvers/adadelta_solver.o\
		./build/solvers/adagrad_solver.o\
		./build/solvers/adam_solver.o\
		./build/solvers/nesterov_solver.o\
		./build/solvers/rmsprop_solver.o\
		./build/solvers/sgd_solver.o\
		./build/util/benchmark.o\
		./build/solver.o

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
		./build/net.o\
		./build/solvers/adadelta_solver.o\
		./build/solvers/adagrad_solver.o\
		./build/solvers/adam_solver.o\
		./build/solvers/nesterov_solver.o\
		./build/solvers/rmsprop_solver.o\
		./build/solvers/sgd_solver.o\
		./build/util/benchmark.o\
		./build/solver.o\
		./build/test/test_lstm_layer.o\
		./build/test/test_caffe_main.o
		

TEST_sw_OBJ=./build/blob.o ./build/common.o ./build/syncedmem.o ./build/layer_factory.o\
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
		./build/glog/logging.o\
		./build/solver.o\
		./build/test/test_caffe_main.o\
		./swtest/obj/test_convolution_layer.o
#		./swtest/obj/conv_layer_impl.o



OBJ_INTEL=./build/common.o \
		./build/util/io.o \
		./build/util/upgrade_proto.o



all: test_solver

test_solver: test_solver.o $(OBJ)
	g++ $^ $(LDFLAGS)  -o $@
test_solver.o: test_solver.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@

vggnet: vggnet.o $(OBJ)
	g++ $^ $(LDFLAGS)  -o $@
vggnet.o: vggnet.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@

test_all: $(TEST_OBJ)
	g++ -pthread $^ ../thirdparty/googletest/libgtest.a $(LDFLAGS) -o $@

test_sw: $(TEST_sw_OBJ)
	g++ $^ ../thirdparty/googletest/libgtest.a $(LDFLAGS) -o $@

test_lstm: test_lstm.o $(OBJ)
	g++ $^ $(LDFLAGS)  -o $@ 
test_lstm.o: test_lstm.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@


net_param_serialize: net_param_serialize.o $(OBJ_INTEL)
	g++ $^ $(LDFLAGS)  -o $@ `pkg-config --libs protobuf`
net_param_serialize.o: net_param_serialize.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@ `pkg-config --cflags protobuf`
caffe.pb.o: ./include/caffe/proto/caffe.pb.cc
	g++ -c $^ $(FLAGS) $(INC_FLAGS) `pkg-config --cflags protobuf`


./swtest/obj/%.o: ./swtest/src/%.cpp
	g++ -c $^ $(FLAGS) $(TEST_INC_FLAGS) -I ./swtest/include -o $@
./build/test/%.o: ./src/test/%.cpp
	g++ -c $^ $(FLAGS) $(TEST_INC_FLAGS) -o $@
./build/%.o: ./src/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/layers/%.o: ./src/layers/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/util/%.o: ./src/util/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/solvers/%.o: ./src/solvers/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/glog/%.o: ./src/glog/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
clean:
	rm *.o ./build/*.o ./build/layers/*.o ./build/util/*.o ./build/solvers/*.o ./build/test/*.o test testcp test_solver test_all
swclean:
	rm swtest/obj/* && rm test_sw

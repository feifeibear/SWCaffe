FLAGS=-O3
FLAGS+=-DCPU_ONLY

INC_FLAGS=-I../thirdparty/glog_install/include -I../thirdparty/openblas_install/include -I./include

OBJ=./build/blob.o ./build/common.o ./build/syncedmem.o ./build/layer_factory.o\
		./build/util/math_function.o \
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
		./build/net.o


test: test.o $(OBJ)
	g++ $^ -L ../thirdparty/glog_install/lib/ -L ../thirdparty/openblas_install/lib -lglog -lopenblas -o $@
test.o: test.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@

testcp: testcp.o $(OBJ)
	g++ $^ -L ../thirdparty/glog_install/lib/ -L ../thirdparty/openblas_install/lib -lglog -lopenblas -o $@
testcp.o: test_conv_pool.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@

./build/%.o: ./src/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/layers/%.o: ./src/layers/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/util/math_function.o: ./src/util/math_functions.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
clean:
	rm *.o ./build/*.o ./build/layers/*.o ./build/util/*.o test

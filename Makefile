FLAGS=-O3
FLAGS+=-DCPU_ONLY

INC_FLAGS=-I./glog_install/include -I./include

OBJ=./build/blob.o ./build/common.o ./build/syncedmem.o\
		./build/util/math_function.o \
		./build/util/insert_splits.o \
		./build/layers/inner_product_layer.o \
		./build/layers/input_layer.o \
		./build/net.o


test: test.o $(OBJ)
	g++ $^ -L ./glog_install/lib/ -lglog -o $@
test.o: test.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/%.o: ./src/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/layers/%.o: ./src/layers/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/util/math_function.o: ./src/util/math_functions.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
clean:
	rm *.o ./build/*.o ./build/layers/*.o ./build/util/*.o test

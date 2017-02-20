FLAGS=-O3
FLAGS+=-DCPU_ONLY

INC_FLAGS=-I./glog_install/include -I./include

OBJ=./build/blob.o ./build/common.o ./build/syncedmem.o\
		./build/util/math_function.o

test: test.o $(OBJ) 
	g++ $^ ./glog_install/lib/libglog.a -o $@
test.o: test.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/%.o: ./src/%.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@
./build/util/math_function.o: ./src/util/math_functions.cpp
	g++ -c $^ $(FLAGS) $(INC_FLAGS) -o $@ 
clean:
	rm *.o ./build/* test 

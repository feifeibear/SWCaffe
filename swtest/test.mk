
ATHREAD_OBJ = ./swtest/obj/athread_test.o\
							./build/swlayers/sw_conv_layer_impl.o\
							./build/swlayers/sw_slave_conv_valid.o\
							./build/swlayers/sw_slave_conv_full.o\
							./build/swlayers/gemm_asm.o

./swtest/obj/athread_test.o: ./swtest/src/athread_test.c 
	mpiCC -host $(FLAGS) $(SWINC_FLAGS) -c $^ -o $@ 



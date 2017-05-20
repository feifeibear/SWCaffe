CXX 	=  	mpiCC -host
LINK 	= 	mpiCC
SWCXX = 	sw5cc.new -slave
FLAGS = 	-O2 -OPT:IEEE_arith=2
#FLAGS +=  -OPT:Olimit=6020
FLAGS += 	-DCPU_ONLY
FLAGS +=  -DMYMPI
#-DSW_CODE 

SWBUILD_DIR=./swbuild
THIRD_PARTY_DIR=../thirdparty
BOOST_DIR=/home/export/online1/swyf/swdnn/softs/install
SWINC_FLAGS=-I./include -I$(THIRD_PARTY_DIR)/CBLAS/include -I${BOOST_DIR}

SWLIBOBJ=$(THIRD_PARTY_DIR)/swblas/SWCBLAS/lib/cblas_LINUX0324.a
SWLIBOBJ+=$(THIRD_PARTY_DIR)/swblas/SWCBLAS/libswblas0324.a

src=$(wildcard ./src/*.cpp ./src/layers/*.cpp ./src/solvers/*.cpp ./src/util/*.cpp ./src/glog/*.cpp)
SWOBJ=$(patsubst ./src/%, $(SWBUILD_DIR)/%, $(patsubst %.cpp, %.o, $(src)))


all: mk test_solver

mk:
	mkdir -p $(SWBUILD_DIR) $(SWBUILD_DIR)/util $(SWBUILD_DIR)/layers $(SWBUILD_DIR)/swlayers \
		$(SWBUILD_DIR)/solvers $(SWBUILD_DIR)/glog 

run:
	bsub -b -I -m 1 -p -q q_sw_share -host_stack 2048 -share_size 5000 -n 1 -cgsp 64 ./vggnet
runlenet:
	bsub -b -I -m 1 -p -q q_sw_share -host_stack 1024 -share_size 6000 -n 1 -cgsp 64 ./test_solver

vggnet: ./models/swobj/vggnet.o $(SWOBJ) $(SWLIBOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@
./models/swobj/vggnet.o: ./models/src/vggnet.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

test_solver: ./models/swobj/test_solver.o $(SWOBJ) $(SWLIBOBJ)
	$(LINK) $^ $(LDFLAGS)  -o $@
./models/swobj/test_solver.o: ./models/src/test_solver.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

$(SWBUILD_DIR)/util/swmatrix_trans.o: ./src/util/swmaxtrix_trans.c
	sw5cc.new -slave -msimd $(FLAGS) $(SWINC_FLAGS) -c $< -o $@
$(SWBUILD_DIR)/util/matrix_trans.o: ./src/util/matrix_trans.c
	sw5cc.new -host -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/swlayers/sw_conv_layer_impl.o: ./src/swlayers/sw_conv_layer_impl.c
	sw5cc.new -host $(FLAGS) $(SWINC_FLAGS) -c $< -o $@
$(SWBUILD_DIR)/swlayers/sw_slave_conv_valid.o: ./src/swlayers/sw_slave_conv_valid.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/swlayers/sw_slave_conv_full.o: ./src/swlayers/sw_slave_conv_full.c
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@
$(SWBUILD_DIR)/swlayers/gemm_asm.o: ./src/swlayers/gemm.S
	$(SWCXX) $(FLAGS) $(SWINC_FLAGS) -msimd -c $< -o $@

$(SWBUILD_DIR)/%.o: ./src/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/layers/%.o: ./src/layers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/util/%.o: ./src/util/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/solvers/%.o: ./src/solvers/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@
$(SWBUILD_DIR)/glog/%.o: ./src/glog/%.cpp
	$(CXX) -c $^ $(FLAGS) $(SWINC_FLAGS) -o $@

clean:
	rm -f $(SWOBJ) vggnet test_solver ./models/swobj/* core*

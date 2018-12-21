#ALL_SHARE=-sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 

mkdir -p ./log && bsub -b -I -J sw_alex_8 -q q_sw_yfb -host_stack 1024 -N 1 -cgsp 64 -sw3run ../../sw3run-all -sw3runarg "-a 1" -cross_size 28000 ../../bin/caffe_sw train --solver=solver_debug.prototxt 2>&1 | tee ./log/sw_alex_mpi_8.log
#mkdir -p ./log && bsub -b -I -q q_sw_yfb -host_stack 1024 -N $1 -np 1 -cgsp 64 -share_size 6000 ./bin/caffe_sw train --solver=models/alexnet/solver_debug.prototxt 2>&1 | tee ./log/sw_alex_`date '+%y-%m-%d-%H-%M-%S'`.log

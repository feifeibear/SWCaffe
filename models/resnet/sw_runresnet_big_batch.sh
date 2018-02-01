#ALL_SHARE=-sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 

mkdir -p ./log && bsub -b -J sw_big_batch_resnet -o ./log/sw_resnet_`date '+%y-%m-%d-%H-%M-%S'`.log -q q_sw_share -host_stack 1024  -N $1 -np 1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./bin/caffe_sw train --solver=models/resnet/solver_debug_big_batch.prototxt
#mkdir -p ./log && bsub -b -I -q q_sw_yfb -host_stack 1024 -N $1 -np 1 -cgsp 64 -share_size 6000 ./bin/caffe_sw train --solver=models/alexnet/solver_debug.prototxt 2>&1 | tee ./log/sw_alex_`date '+%y-%m-%d-%H-%M-%S'`.log

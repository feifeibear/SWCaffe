#mkdir -p ./log && bsub -b -I -J sw_googlenet_debug -q q_sw_yfb -host_stack 1024 -N $1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./bin/caffe_sw train --solver=models/googlenet/quick_solver_debug.prototxt 2>&1 | tee ./log/sw_googlenet_debug_`date '+%y-%m-%d-%H-%M-%S'`.log
#mkdir -p ./log && bsub -b -I -q q_sw_yfb -host_stack 1024 -N $1 -np 1 -cgsp 64 -share_size 6000 ./bin/caffe_sw train --solver=models/vgg/solver.prototxt 2>&1 | tee ./log/sw_vgg_`date '+%y-%m-%d-%H-%M-%S'`.log
for((nodenum = 32;nodenum <= 128;nodenum = nodenum * 2))
{
mkdir -p ./log && bsub -b -I -J sw_googlenet_zcj -q q_sw_share -host_stack 256 -N $nodenum -np 1 -cgsp 64 -sw3run ../../sw3run-all -sw3runarg "-a 1" -cross_size 29000 ../../bin/caffe_sw train --solver=quick_solver_debug.prototxt 2>&1 | tee ./log/sw_googlenet_debug_$nodenum.log
}

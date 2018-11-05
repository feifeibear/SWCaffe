#ALL_SHARE=-sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 

#mkdir -p ./log && bsub -b -I -q q_x86_vio_yfb -N $1 ./build/tools/caffe time --model=models/lstm/basiclstm.prototxt -gpu 0 2>&1 | tee ./log/vio_lstm_`date '+%y-%m-%d-%H-%M-%S'`.log
mkdir -p ./log && bsub -b -I -J sw_lstm_debug -q q_sw_yfb -host_stack 1024 -N 1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./bin/caffe_sw time --model=models/lstm/basiclstm.prototxt 2>&1 | tee ./log/sw_lstm_`date '+%y-%m-%d-%H-%M-%S'`.log
#mkdir -p ./log && bsub -b -I -q q_sw_yfb -host_stack 1024 -N $1 -np 1 -cgsp 64 -share_size 6000 ./bin/caffe_sw train --solver=models/alexnet/solver_debug.prototxt 2>&1 | tee ./log/sw_alex_`date '+%y-%m-%d-%H-%M-%S'`.log

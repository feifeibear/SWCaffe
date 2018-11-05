#mkdir -p ./log && bsub -b -I -o ./log/sw_alexnet_big_batch_64k.log -J lstm -q q_sw_yfb -host_stack 1024 -N 1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./bin/lstm_sequence ./models/lstm_sequence/lstm_short_solver.prototxt lstm_short_result.log 320

mkdir -p ./log && bsub -b -I -J sw_lstm_debug -q q_sw_yfb -host_stack 1024 -N 1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./bin/caffe_sw train --solver=models/lstm/lstm_short_solver.prototxt 2>&1 | tee ./log/sw_lstm_`date '+%y-%m-%d-%H-%M-%S'`.log

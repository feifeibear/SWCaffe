#ALL_SHARE=-sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 
#mkdir -p ./log && bsub -b -I -m 1 -q q_sw_share -host_stack 500 -share_size 6500 -n $1 -cgsp 64 ./bin/alexnet_sw 2>&1 | tee ./log/sw_alex_`date '+%y-%m-%d-%H-%M-%S'`.log

mkdir -p ./log && bsub -b -I -q q_sw_expr -host_stack 1024 -n $1 -cgsp 64 -sw3run ./sw3run-all -sw3runarg "-a 1" -cross_size 28000 ./bin/alexnet_sw 2>&1 | tee ./log/sw_alex_`date '+%y-%m-%d-%H-%M-%S'`.log

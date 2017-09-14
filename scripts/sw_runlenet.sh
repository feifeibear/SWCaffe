mkdir -p ./log && bsub -b -I -m 1 -p -q q_sw_share -host_stack 1024 -share_size 6000 -n $1 -cgsp 64 ./bin/test_lenet_sw 2>&1 | tee ./log/sw_lenet_`date '+%y-%m-%d-%H-%M-%S'`.log

source ./scripts/env.sh
mkdir -p ./log && bsub -I -q q_x86_vio_yfb -n $1 ./bin/test_lenet_intel 2>&1 | tee ./log/intel_lenet_`date '+%y-%m-%d-%H-%M-%S'`.log

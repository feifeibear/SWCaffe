source ./scripts/env.sh
mkdir -p ./log && bsub -I -q q_x86_vio_yfb -n $1 ./bin/test_lstm_intel 2>&1 | tee ./log/intel_lstm_`date '+%y-%m-%d-%H-%M-%S'`.log

source ./scripts/env.sh
mkdir -p ./log && bsub -I -q q_x86_expr -n $1 ./bin/vggnet_intel 2>&1 | tee ./log/intel_vgg_`date '+%y-%m-%d-%H-%M-%S'`.log

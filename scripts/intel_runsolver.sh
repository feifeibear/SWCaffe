source ./scripts/env.sh
mkdir -p ./log && bsub -I -q q_x86_expr -n $1 ./bin/test_solver_intel 2>&1 | tee ./log/intel_solver_`date '+%y-%m-%d-%H-%M-%S'`.log

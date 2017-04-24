ps -e | grep test_solver | awk '{print $1}' | xargs kill -9
ps -e | grep test_solver

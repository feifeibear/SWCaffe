export LD_LIBRARY_PATH=/home/fang/Code/thirdparty/openblas_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/fang/Code/thirdparty/glog_install/lib:$LD_LIBRARY_PATH
mpirun -n 4 ./vggnet

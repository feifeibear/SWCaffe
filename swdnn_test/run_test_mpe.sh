export LD_LIBRARY_PATH=../../thirdparty/openblas_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=../../thirdparty/glog_install/lib:$LD_LIBRARY_PATH
export MV2_ENABLE_AFFINITY=0
./bin/caffe_main
#export OPENBLAS_NUM_THREADS=24
#export OMP_NUM_THREADS=24

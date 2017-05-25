ROOT_DIR=/home/export/online1/swyf/swdnn/fjr/SWCaffe/thirdparty
export LD_LIBRARY_PATH=$ROOT_DIR/openblas_install/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ROOT_DIR/glog_install/lib:$LD_LIBRARY_PATH
rm -rf alex.log
bsub -I -q q_x86_vio_yfb -n 1 ./alexnet_intel 2>&1 | tee ture-alex16.log

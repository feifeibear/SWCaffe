SRC=caffe.proto
bsub -I -b -q q_sw_yfb -n 1 -cgsp 64 -host_stack 1024 -share_size 4096 ../../../../thirdparty/bin/protoc $SRC --cpp_out=.

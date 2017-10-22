/*
 * Edited by zhaowl
 * 2017/10/22
*/

This is a distributed Caffe on Sunway TaihuLight Supercomputer (sunwayCaffe).

How to use:
====Simple Use====
1. Make sure the dependencies are at ../thirdparty/
2. At $(YourPath)=*/sunwayCaffe run:
    make -j caffe
3. Run scripts: see models/ for examples
    sh ./models/vgg/sw_runvgg.sh
  * change -N x to run distributed training with 1 server and (x-1) workers

*4CG parallel is not supported in this version. 

===For non-distributed version===
Uncomment the compilation flag -USE_SWMPI and recompile the project, and use "-N 1" for bsub command in the scripts.

===Debug Options===
Use the compilation flags (USE_*/DEBUG_VERBOSE_*) in Makefile to print debug/profile info. See exaplanations in Makefile.



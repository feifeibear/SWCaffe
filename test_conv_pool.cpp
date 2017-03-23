#include "caffe/caffe.hpp"
using namespace caffe;

int main () {
  ConvolutionParameter cp;
  cp.add_kernel_size(3);
  return 0;
}

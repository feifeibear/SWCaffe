#include "caffe/caffe.hpp"
using namespace caffe;

int main () {
  NetParameter net_param;
  Net<float> net(net_param);

  return 0;
}

//#include <caffe/protohpp/InnerProductParameter.hpp>
//#include <caffe/protohpp/InputParameter.hpp>
//#include <caffe/protohpp/ConvolutionParameter.hpp>
//#include <caffe/protohpp/PoolingParameter.hpp>
//#include <caffe/protohpp/DataParameter.hpp>
//#include <caffe/protohpp/ReLUParameter.hpp>
#include <caffe/protohpp/SoftmaxParameter.hpp>
#include <caffe/protohpp/LossParameter.hpp>
#include <caffe/protohpp/AccuracyParameter.hpp>

namespace caffe{

SoftmaxParameter SoftmaxParameter::default_instance_;
LossParameter LossParameter::default_instance_;
AccuracyParameter AccuracyParameter::default_instance_;

}
//#include <caffe/protohpp/InnerProductParameter.hpp>
//#include <caffe/protohpp/InputParameter.hpp>
//#include <caffe/protohpp/ConvolutionParameter.hpp>
//#include <caffe/protohpp/PoolingParameter.hpp>
//#include <caffe/protohpp/DataParameter.hpp>
//#include <caffe/protohpp/ReLUParameter.hpp>
#include <caffe/protohpp/SoftmaxParameter.hpp>
#include <caffe/protohpp/LossParameter.hpp>
#include <caffe/protohpp/AccuracyParameter.hpp>
#include <caffe/protohpp/ScaleParameter.hpp>
#include <caffe/protohpp/ConcatParameter.hpp>
#include <caffe/protohpp/BiasParameter.hpp>
#include <caffe/protohpp/SliceParameter.hpp>
#include <caffe/protohpp/ReductionParameter.hpp>

namespace caffe{

SoftmaxParameter SoftmaxParameter::default_instance_;
LossParameter LossParameter::default_instance_;
AccuracyParameter AccuracyParameter::default_instance_;
ScaleParameter ScaleParameter::default_instance_;
ConcatParameter ConcatParameter::default_instance_;
BiasParameter BiasParameter::default_instance_;
SliceParameter SliceParameter::default_instance_;
ReductionParameter ReductionParameter::default_instance_;

}
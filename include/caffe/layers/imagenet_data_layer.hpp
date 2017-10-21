#ifndef CAFFE_IMAGENET_DATA_LAYER_HPP_
#define CAFFE_IMAGENET_DATA_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/db.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class IMAGENETDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit IMAGENETDataLayer(const LayerParameter& param);
  virtual ~IMAGENETDataLayer();
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "IMAGENETDataLayer"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  void Next();
  bool Skip();
  void load_batch(Batch<Dtype>* batch);
  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  shared_ptr<Batch<Dtype> > fetch_;
  Blob<Dtype> transformed_data_;
  uint64_t offset_;

};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_

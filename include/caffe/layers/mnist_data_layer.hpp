#ifndef CAFFE_MNIST_DATA_LAYER_HPP_
#define CAFFE_MNIST_DATA_LAYER_HPP_

#include "caffe/layers/data_layer.hpp"

namespace caffe {

template <typename Dtype>
class MNISTDataLayer : public DataLayer<Dtype> {
public:
  explicit MNISTDataLayer(const LayerParameter& param);
  virtual ~MNISTDataLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "MNISTData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

protected:
 int reverseInt (int i) {
      unsigned char c1, c2, c3, c4;
      c1 = i & 255;
      c2 = (i >> 8) & 255;
      c3 = (i >> 16) & 255;
      c4 = (i >> 24) & 255;
      return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  }
  int n_cols;
  int n_rows;
  int number_of_images;
  int batch_size;
  std::ifstream file;
  std::ifstream label_file;
  //void Next();
  //bool Skip();
  //virtual void load_batch(Batch<Dtype>* batch);

  //shared_ptr<db::DB> db_;
  //shared_ptr<db::Cursor> cursor_;
  long offset_;

  bool start;
};

}
#endif

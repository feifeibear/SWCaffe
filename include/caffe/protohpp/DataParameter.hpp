#ifndef _DATAPARAMETER_
#define _DATAPARAMETER_ 

namespace caffe {

class DataParameter {

  public:
    DataParameter(){}
    explicit DataParameter(int B):batch_size_(B){}
    inline void CopyFrom(const DataParameter& other) {
      batch_size_ = other.batch_size();
    }
    inline int batch_size() const { return batch_size_; }
    inline void set_batch_size(int B) { batch_size_ = B; }

  private:
    int batch_size_;
};
}

#endif

#ifndef _DUMMY_DATA_PARAMETER_
#define _DUMMY_DATA_PARAMETER_

#include "FillerParameter.hpp"

namespace caffe {

  class DummyDataParameter {
    public:
      DummyDataParameter() {}
      ~DummyDataParameter() {}

      inline DummyDataParameter& operator=( const DummyDataParameter& other ) {
        this->CopyFrom(other);
        return *this;
      }
      inline void CopyFrom( const DummyDataParameter& other ) {
        int data_filler_size = other.data_filler().size();
        for( int i = 0; i < data_filler_size; ++i )
          data_filler_.push_back(other.data_filler()[i]);
        int shape_size = other.shape().size();
        for( int i = 0; i < shape_size; ++i )
          shape_.push_back(other.shape()[i]);

        for(int i = 0; i < other.num_size();++i)
          num_.push_back(other.num(i));
        for(int i = 0; i < other.channels_size();++i)
          channels_.push_back(other.channels(i));

        for(int i = 0; i < other.height_size();++i)
          height_.push_back(other.height(i));

        for(int i = 0; i < other.width_size();++i)
          width_.push_back(other.width(i));

      }

      inline int shape_size() const { return shape_.size(); }
      inline const std::vector<BlobShape>& shape() const { return shape_; }
      inline const BlobShape& shape(int idx) const { return shape_[idx]; }
      inline BlobShape* add_shape() {
        BlobShape b;
        shape_.push_back(b);
        return &shape_[shape_.size()-1];
      }

      inline int data_filler_size() const { return data_filler_.size(); }
      inline const std::vector<FillerParameter>& data_filler() const {
        return data_filler_;
      }
      inline const FillerParameter data_filler(int idx) const {
        return data_filler_[idx];
      }
      inline FillerParameter* add_data_filler() {
        FillerParameter f;
        data_filler_.push_back(f);
        return &data_filler_[data_filler_.size()-1];
      }
      inline FillerParameter* mutable_data_filler() {
        return &data_filler_[0];
      }
      inline FillerParameter* mutable_data_filler(int idx) {
        return &data_filler_[idx];
      }

      inline int num_size() const { return num_.size(); }
      inline int channels_size() const { return channels_.size(); }
      inline int height_size() const { return height_.size(); }
      inline int width_size() const { return width_.size(); }

      inline int height(int idx) const {return height_[idx];}
      inline int num(int idx) const {return num_[idx];}
      inline int channels(int idx) const {return channels_[idx];}
      inline int width(int idx) const {return width_[idx];}

      inline void add_height(int n) { height_.push_back(n); }
      inline void add_num(int n) { num_.push_back(n); }
      inline void add_width(int n) { width_.push_back(n); }
      inline void add_channels(int n) { channels_.push_back(n); }

    private:
      std::vector<FillerParameter> data_filler_;
      std::vector<BlobShape> shape_;
      std::vector<int> num_;
      std::vector<int> channels_;
      std::vector<int> height_;
      std::vector<int> width_;
  };
}
#endif

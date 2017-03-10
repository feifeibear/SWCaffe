#ifndef _NETPARAMETER_H_
#define _NETPARAMETER_H_

/**
 * rewrite from caffe.pb.h
 */

#include "caffe/LayerParameter.hpp"
#include <vector>
#include <string>

namespace caffe {

class NetState
{
public:
  NetState(){
    stage_.resize(0);
    level_ = 0;
  }
  virtual ~NetState(){
    phase_ = TRAIN;
  }
  NetState(const NetState& from) {
    CopyFrom(from);
  }

  int stage_size() const { return stage_.size(); }
  std::string stage( int id ) const { return stage_[id]; }
  const std::vector<std::string> get_stage() const { return stage_; }

  inline std::string name() const { return name_; } 

  inline Phase phase() const { return phase_; }

  int level() const { return level_; }

  void CopyFrom(const NetState& other) {
    name_ = other.name();
    phase_ = other.phase();
    level_ = other.level();
    stage_.clear();
    int stage_size = other.get_stage().size();
    for( int i = 0; i < stage_size; ++i ){
      stage_.push_back(other.get_stage()[i]);
    }
  }

private:
  std::string name_;
  Phase phase_;
  int level_;
  std::vector<std::string> stage_;
};

class NetParameter {
  public:
    explicit NetParameter(std::string name, std::vector<LayerParameter> layerparams):
      name_(name), layer_(layerparams), force_backward_(true), debug_info_(false) {}
    NetParameter(){}

    ~NetParameter() {}
    inline std::string name() const { return name_; }
    inline NetState state() const { return state_; }

    inline int layer_size() const { return layer_.size();  }
    LayerParameter layer(int id) const { return layer_[id]; }
    const std::vector<LayerParameter> get_layer() const { return layer_; }

    LayerParameter* mutable_layer(int id) { return &layer_[id]; }
    inline bool force_backward() const { return force_backward_; }
    inline bool debug_info() const { return debug_info_; }
    std::string DebugString() const { return "TODO in NetParameter.hpp DebugString"; }

    //TODO wait for a better implementation
    void CopyFrom(const NetParameter& other) {
      this->name_ = other.name();
      force_backward_ = other.force_backward();
      debug_info_ = other.debug_info();
      state_.CopyFrom(other.state());
      int layer_size = other.get_layer().size();
      for( int i = 0; i < layer_size; ++i )
        layer_.push_back(other.get_layer()[i]);
    }

    void clear_layer() {
      layer_.clear();
    }
    //TODO
    LayerParameter* add_layer() {
      layer_.resize(layer_.size()+1);
      //LayerParameter* new_layer = new LayerParameter();
      //layer_.push_back(*new_layer);
      //return new_layer;
      return &layer_[layer_.size()-1];
    }

  private:
    std::string name_;
    bool force_backward_;
    bool debug_info_;

    NetState state_;
    std::vector<LayerParameter> layer_;
};

}

#endif

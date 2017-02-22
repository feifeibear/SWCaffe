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
  NetState(){}
  virtual ~NetState(){}
  NetState(const NetState& from) {}

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
    stage_.assign( other.get_stage().begin(), other.get_stage().end() );
  }

private:
  std::string name_;
  Phase phase_;
  int level_;
  std::vector<std::string> stage_;
};

class NetParameter {
  public:
    NetParameter() {}
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
      layer_.assign(other.get_layer().begin(), other.get_layer().end());
    }

    void clear_layer() {
      layer_.clear();
    }
    //TODO
    LayerParameter* add_layer() {
      LayerParameter* new_layer = new LayerParameter();
      layer_.push_back(*new_layer);
      return new_layer;
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

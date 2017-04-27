#ifndef _SERIALIZE_HPP_
#define _SERIALIZE_HPP_

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

class Serial_Blob
{
public:
  std::vector<float> data;
  std::vector<float> diff;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & data;
    ar & diff;
  }
};

class Serial_Layer
{
public:
  std::vector<Serial_Blob> blobs;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & blobs;
  }
};

class Serial_Net
{
public:
  std::vector<Serial_Layer> layers;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & layers;
  }
};

#endif
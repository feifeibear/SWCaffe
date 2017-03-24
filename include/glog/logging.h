#pragma once

#include <iostream>

#define LOG(_cond)  \
  std::cout

#define LOG_IF(val1, _cond)  \
  if(false) std::cout

#define DLOG(_cond)  \
  std::cout

#define CHECK_LE(val1, val2) \
  if(false) std::cout  

#define CHECK_GE(val1, val2) \
  if(false) std::cout  

#define DCHECK_GE(val1, val2) \
  if(false) std::cout  

#define CHECK_EQ(val1, val2) \
  if(false) std::cout  

#define CHECK(cond_) \
  if(false) std::cout  

#define DCHECK(cond_) \
  if(false) std::cout  

#define CHECK_GT(val1, val2) \
  if(false) std::cout  

#define DCHECK_GT(val1, val2) \
  if(false) std::cout  

#define CHECK_LT(val1, val2) \
  if(false) std::cout  

#define DCHECK_LT(val1, val2) \
  if(false) std::cout  

#define CHECK_NE(val1, val2) \
  if(false) std::cout  

//#define CHECK_NOTNULL(val)

#define LOG_FIRST_N(val1, val2)\
  if(false) std::cout  

#define CHECK_NOTNULL(val)\
  std::cout

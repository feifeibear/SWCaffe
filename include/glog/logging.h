#ifndef __LOG_H__
#define __LOG_H__

#include <sstream>
#include <string>
#include <cstdio>
#include <boost/format.hpp>

using boost::format;

std::string getDateTime();

enum TLogLevel {logERROR, logWARNING, logINFO, logDEBUG, logDEBUG1, logDEBUG2, logDEBUG3, logDEBUG4, logTRACE};

template <typename T>
class Log
{
public:
  Log();
  virtual ~Log();
  std::ostringstream& Get(TLogLevel level = logINFO);
public:
  static void setLogFile(const std::string &file);
  static TLogLevel& ReportingLevel();
  static std::string ToString(TLogLevel level);
  static TLogLevel FromString(const std::string& level);
protected:
  std::ostringstream os;
private:
  Log(const Log&);
  Log& operator =(const Log&);
};

template <typename T>
Log<T>::Log()
{
}

template <typename T>
std::ostringstream& Log<T>::Get(TLogLevel level)
{
  os << getDateTime();
  os << " " << ToString(level) << ": ";
  os << std::string(level > logDEBUG ? level - logDEBUG : 0, '\t');
  return os;
}

template <typename T>
Log<T>::~Log()
{
  os << std::endl;
  T::Output(os.str());
}

template <typename T>
TLogLevel& Log<T>::ReportingLevel()
{
  static TLogLevel reportingLevel = logDEBUG4;
  return reportingLevel;
}

template <typename T>
std::string Log<T>::ToString(TLogLevel level)
{
  static const char* const buffer[] = {"ERROR", "WARNING", "INFO", "DEBUG", "DEBUG1", "DEBUG2", "DEBUG3", "DEBUG4", "TRACE"};
  return buffer[level];
}

template<typename T>
inline void Log<T>::setLogFile(const std::string& file) {
  T::setFileName(file);
}

template <typename T>
TLogLevel Log<T>::FromString(const std::string& level)
{
  if (level == "DEBUG4")
    return logDEBUG4;
  if (level == "DEBUG3")
    return logDEBUG3;
  if (level == "DEBUG2")
    return logDEBUG2;
  if (level == "DEBUG1")
    return logDEBUG1;
  if (level == "DEBUG")
    return logDEBUG;
  if (level == "INFO")
    return logINFO;
  if (level == "WARNING")
    return logWARNING;
  if (level == "ERROR")
    return logERROR;
  if (level == "TRACE")
    return logTRACE;
  Log<T>().Get(logWARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
  return logINFO;
}

class Output2FILE
{
public:
  static void Output(const std::string& msg);
  static void setFileName(const std::string &filename);
  static FILE*& TerminalStream();

private:
  static FILE*& FileStream();
  static FILE *pfile;
  static std::string filename;
};

class FILELog: public Log<Output2FILE> {
};


#ifndef FILELOG_MAX_LEVEL
#define FILELOG_MAX_LEVEL logDEBUG4
#endif

#define FILE_LOG(level) \
    FILELog().Get(level)

//#define FILE_LOG(level) \
//    if (level > FILELOG_MAX_LEVEL){}\
//    else if (level > FILELog::ReportingLevel() || !Output2FILE::TerminalStream()) {} \
//    else FILELog().Get(level)

//#define TRACE() FILE_LOG(logDEBUG1)
//#define DEBUG() FILE_LOG(logDEBUG)
//#define INFO() FILE_LOG(logINFO)
//#define WARNING() FILE_LOG(logWARNING)
//#define ERROR() FILE_LOG(logERROR)

#define LOG(_cond)  \
  FILE_LOG(logINFO) << __FILE__ << " : line " << __LINE__<< " : "

#define LOG_IF(val1, _cond)  \
  if(_cond) \
   FILE_LOG(logINFO) << __FILE__ << " : line " << __LINE__<< " : "


#define DLOG(_cond)  \
  FILE_LOG(logINFO) << __FILE__<< " : line " << __LINE__<< " : "

#define CHECK_LE(val1, val2) \
  if(!(val1 <= val2)) \
  FILE_LOG(logINFO) << __FILE__<< " : line " << __LINE__<< " : "


#define CHECK_GE(val1, val2) \
  if(!(val1 >= val2)) std::cout  

#define DCHECK_GE(val1, val2) \
  if(!(val1 >= val2)) std::cout  

#define CHECK_EQ(val1, val2) \
  if(!(val1 == val2)) std::cout  

#define CHECK(cond_) \
  if(!(cond_)) std::cout  

#define DCHECK(cond_) \
  if(! (cond_)) std::cout  

#define CHECK_GT(val1, val2) \
  if(!(val1 > val2)) std::cout  

#define DCHECK_GT(val1, val2) \
  if(!(val1 > val2)) std::cout  

#define CHECK_LT(val1, val2) \
  if(!(val1 < val2)) std::cout  

#define DCHECK_LT(val1, val2) \
  if(!(val1 < val2)) std::cout  

#define CHECK_NE(val1, val2) \
  if(val1 == val2) std::cout  

//#define CHECK_NOTNULL(val)

#define LOG_FIRST_N(val1, val2)\
  if(false) std::cout  

#define CHECK_NOTNULL(val)\
  std::cout

#endif //__LOG_H__

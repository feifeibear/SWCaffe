/*
 * logger.cpp
 *
 *  Created on: Apr 7, 2016
 *      Author: rice
 */

#include "glog/logging.h"

#include <sys/time.h>

FILE *Output2FILE::pfile = NULL;
std::string Output2FILE::filename = "";

std::string getDateTime()
{
  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
  // for more information about date/time format
  strftime(buf, sizeof(buf), "%Y-%m-%d %X %s ", &tstruct);

  return buf;
}

void Output2FILE::setFileName(const std::string& _filename) {
  filename = _filename;
}

FILE*& Output2FILE::FileStream() {
  static bool init = false;
  if (!init) {
    init = true;
    pfile = fopen(filename.c_str(), "w");
  }

  return pfile;
}

FILE*& Output2FILE::TerminalStream()
{
    static FILE* pStream = stdout;
    return pStream;
}

void Output2FILE::Output(const std::string& msg)
{
    FILE* tstream = TerminalStream();
    FILE *fstream = FileStream();
    if (!tstream)
        return;
    fprintf(tstream, "%s", msg.c_str());
    fflush(tstream);

    if (fstream != NULL) {
      fprintf(fstream, "%s", msg.c_str());
      fflush(fstream);
    }
}

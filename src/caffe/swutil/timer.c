#include "caffe/util/timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#define MAXTIMERSIZE 1000
int timer[MAXTIMERSIZE];
int timer_num = 0;
int isTiming = 0;
long startTime;
long startSec;
long startUsec;
char* timer_name[MAXTIMERSIZE];
int timer_index;

void begin_timer(const char *fn) {
  if(isTiming) {
    printf("ERROR: only one function can timing at a time.");
    return ;
  }
  if(timer_num>=MAXTIMERSIZE-1) {
    printf("ERROR: too much timer.");
    return ;
  }
  int i;
  int found=0;
  struct timeval tval;
  for(i=0;i<timer_num;i++) {
    if(strcmp(timer_name[i],fn)==0) {
      timer_index = i;
      found = 1;
      break;
    }
  }
  if(!found) {
    timer_name[timer_num] = (char*)malloc(strlen(fn));
    strcpy(timer_name[timer_num],fn);
    timer_index = timer_num;
    ++timer_num;
  }
  isTiming = 1;
  //startTime = clock();
  gettimeofday(&tval,0);
  startTime = tval.tv_sec*1000000+tval.tv_usec;
  startSec = tval.tv_sec;
  startUsec= tval.tv_usec;
}

void stop_timer() {
  struct timeval tval;
  gettimeofday(&tval,0);
  long endTime = tval.tv_sec*1000000+tval.tv_usec;
  if(!isTiming) return ;
  isTiming = 0;
  timer[timer_index]+= (tval.tv_sec - startSec)*1e6 + (tval.tv_usec - startUsec);
}

void print_timer() {
  int i;
  printf("\n");
  for(i=0;i<timer_num;i++) {
    printf("Routine %s time: %lf\n",timer_name[i],((double)timer[i])/1e6);
  }
}

/*************************
 * created by Xin You     
 * Date : 2017/8/3        
 * Description: a timer which can log excute time of functions
 * usage: at beginning: begin_timer("function name");
 *        at endding:   stop_timer();
 *        print all logged timer:  print_timer();
 ************************/

#ifndef TIMER_H_
#define TIMER_H_
void begin_timer(const char* fn);
void stop_timer();
void print_timer();
#endif

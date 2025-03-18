#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <time.h>

#define LOG_LEVEL_NONE  0
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_INFO  2
#define LOG_LEVEL_DEBUG 3

#ifndef CURRENT_LOG_LEVEL
#define CURRENT_LOG_LEVEL LOG_LEVEL_DEBUG
#endif

#define LOG(level, level_str, fmt, ...)                                       \
    do {                                                                      \
        time_t t = time(NULL);                                                \
        struct tm *tm_info = localtime(&t);                                   \
        char time_str[20];                                                    \
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", tm_info);     \
        fprintf(stderr, "[%s] [%s] %s:%d " fmt "\n",                           \
                time_str, level_str, __FILE__, __LINE__, ##__VA_ARGS__);        \
    } while (0)


#if CURRENT_LOG_LEVEL >= LOG_LEVEL_ERROR
#define LOG_ERROR(fmt, ...) LOG(LOG_LEVEL_ERROR, "ERROR", fmt, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) do {} while(0)
#endif

#if CURRENT_LOG_LEVEL >= LOG_LEVEL_INFO
#define LOG_INFO(fmt, ...)  LOG(LOG_LEVEL_INFO, "INFO", fmt, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...)  do {} while(0)
#endif

#if CURRENT_LOG_LEVEL >= LOG_LEVEL_DEBUG
#define LOG_DEBUG(fmt, ...) LOG(LOG_LEVEL_DEBUG, "DEBUG", fmt, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) do {} while(0)
#endif

#endif
//
// Created by 付聪 on 2017/6/21.
//

#ifndef EFANNA2E_UTIL_H
#define EFANNA2E_UTIL_H
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>

#ifdef __APPLE__
#else
#include <malloc.h>
#endif

#ifdef __AVX__
#define DATA_ALIGN_FACTOR 8
#else
#ifdef __SSE2__
#define DATA_ALIGN_FACTOR 4
#else
#define DATA_ALIGN_FACTOR 1
#endif
#endif

namespace efanna2e {

static void GenRandom(std::mt19937 &rng, unsigned *addr, unsigned size,
                      unsigned N) {
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  std::sort(addr, addr + size);
  for (unsigned i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  unsigned off = rng() % N;
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

inline float *data_align(float *data_ori, unsigned point_num, unsigned &dim) {
  return data_ori;

  //   //std::cout << "align with : "<<DATA_ALIGN_FACTOR << std::endl;
  //   float* data_new=0;
  //   unsigned new_dim = (dim + DATA_ALIGN_FACTOR - 1) / DATA_ALIGN_FACTOR *
  //   DATA_ALIGN_FACTOR;
  //   //std::cout << "align to new dim: "<<new_dim << std::endl;
  //   struct alignas(DATA_ALIGN_FACTOR * 4) OverAligned { char b; };
  //   data_new = (float*) new OverAligned[point_num * new_dim * sizeof(float)];

  //   for(unsigned i=0; i<point_num; i++){
  //     memcpy(data_new + i * new_dim, data_ori + i * dim, dim *
  //     sizeof(float)); memset(data_new + i * new_dim + dim, 0, (new_dim - dim)
  //     * sizeof(float));
  //   }
  //   dim = new_dim;
  //   delete[] data_ori;
  //   return data_new;
}

} // namespace efanna2e

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

#if defined(_WIN32)

// windows.h will define max/min as macro, disable this function and import
// limits instead
#define NOMINMAX
#include <limits>
#include <psapi.h>
#include <windows.h>


#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))

#include <sys/resource.h>
#include <unistd.h>


#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  /* AIX and Solaris ------------------------------------------ */
  struct psinfo psinfo;
  int fd = -1;
  if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
    return (size_t)0L; /* Can't open? */
  if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
    close(fd);
    return (size_t)0L; /* Can't read? */
  }
  close(fd);
  return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))
  /* BSD, Linux, and OSX -------------------------------------- */
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
  return (size_t)rusage.ru_maxrss;
#else
  return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
  /* Unknown OS ----------------------------------------------- */
  return (size_t)0L; /* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &infoCount) != KERN_SUCCESS)
    return (size_t)0L; /* Can't access? */
  return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)
  /* Linux ---------------------------------------------------- */
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t)0L; /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t)0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
  /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
  return (size_t)0L; /* Unsupported. */
#endif
}

#endif // EFANNA2E_UTIL_H

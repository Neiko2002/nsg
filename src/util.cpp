#if defined(_WIN32)
    #include <SDKDDKVer.h>
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>

// Manually declare what we need from PSAPI to avoid header conflicts
typedef struct _PROCESS_MEMORY_COUNTERS {
    DWORD cb;
    DWORD PageFaultCount;
    SIZE_T PeakWorkingSetSize;
    SIZE_T WorkingSetSize;
    SIZE_T QuotaPeakPagedPoolUsage;
    SIZE_T QuotaPagedPoolUsage;
    SIZE_T QuotaPeakNonPagedPoolUsage;
    SIZE_T QuotaNonPagedPoolUsage;
    SIZE_T PagefileUsage;
    SIZE_T PeakPagefileUsage;
} PROCESS_MEMORY_COUNTERS;
typedef PROCESS_MEMORY_COUNTERS* PPROCESS_MEMORY_COUNTERS;

extern "C" {
BOOL WINAPI GetProcessMemoryInfo(HANDLE Process, PPROCESS_MEMORY_COUNTERS ppsmemCounters, DWORD cb);
}
    #pragma comment(lib, "psapi.lib")
#endif

#include "efanna2e/util.h"

#if defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    #include <sys/resource.h>
    #include <unistd.h>
    #if defined(__APPLE__) && defined(__MACH__)
        #include <mach/mach.h>
    #elif (defined(_AIX) || defined(__TOS__AIX__)) || \
        (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
        #include <fcntl.h>
        #include <procfs.h>
    #endif
#endif

#include <cstdio>

size_t getPeakRSS() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) return (size_t)info.PeakWorkingSetSize;
    return (size_t)0;
#elif (defined(_AIX) || defined(__TOS__AIX__)) || \
    (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1) return (size_t)0L;
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);
#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    #if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
    #else
    return (size_t)(rusage.ru_maxrss * 1024L);
    #endif
#else
    return (size_t)0L;
#endif
}

size_t getCurrentRSS() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) return (size_t)info.WorkingSetSize;
    return (size_t)0;
#elif defined(__APPLE__) && defined(__MACH__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) != KERN_SUCCESS) return (size_t)0L;
    return (size_t)info.resident_size;
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    long rss = 0L;
    FILE* fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL) return (size_t)0L;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t)0L;
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
#else
    return (size_t)0L;
#endif
}

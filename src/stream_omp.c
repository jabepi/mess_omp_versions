/*
 * Copyright (c) 2024, Barcelona Supercomputing Center
 * Contact: mess             [at] bsc [dot] es
 *          pouya.esmaili    [at] bsc [dot] es
 *          petar.radojkovic [at] bsc [dot] es
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the name of the copyright holder nor the names
 *       of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

# define _XOPEN_SOURCE 600

# include <stdio.h>
# include <stdlib.h>
# include <unistd.h>
# include <math.h>
# include <float.h>
# include <string.h>
# include <limits.h>
# include <sys/time.h>
# include <stdint.h>
# include <omp.h>

#define DEBUG_LOG_PATH "/home/gem5/mess_omp2/debug-2ac992.log"
#define DEBUG_SESSION_ID "2ac992"

static long long debug_now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((long long)tv.tv_sec * 1000LL) + ((long long)tv.tv_usec / 1000LL);
}

static void debug_log_json(const char *run_id,
                           const char *hypothesis_id,
                           const char *location,
                           const char *message,
                           const char *data_json)
{
    fprintf(stdout,
            "{\"sessionId\":\"%s\",\"runId\":\"%s\",\"hypothesisId\":\"%s\","
            "\"location\":\"%s\",\"message\":\"%s\",\"data\":%s,\"timestamp\":%lld}\n",
            DEBUG_SESSION_ID,
            run_id,
            hypothesis_id,
            location,
            message,
            data_json,
            debug_now_ms());
    fflush(stdout);
}

void m5_dump_reset_stats(uint64_t delay, uint64_t period) {
#if defined(__aarch64__)
    register uint64_t x0 __asm__("x0") = delay;
    register uint64_t x1 __asm__("x1") = period;
    __asm__ __volatile__ (
        ".inst 0xFF420110\n\t"
        : "+r" (x0), "+r" (x1)
    );
#endif
}

void m5_exit(uint64_t delay) {
#if defined(__aarch64__)
    register uint64_t x0 __asm__("x0") = delay;
    __asm__ __volatile__ (
        ".inst 0xFF210110\n\t"
        : "+r" (x0)
    );
#endif
}

void m5_dump_stats(uint64_t delay, uint64_t period) {
#if defined(__aarch64__)
    register uint64_t x0 __asm__("x0") = delay;
    register uint64_t x1 __asm__("x1") = period;
    __asm__ __volatile__ (
        ".inst 0xFF410110\n\t"
        : "+r" (x0), "+r" (x1)
    );
#endif
}

# include "utils.h"

/*-----------------------------------------------------------------------
 * The benchmark is based on the modified STREAM benchmark
 * (original STREAM benchmark: http://www.cs.virginia.edu/stream/).
 * Contrary to the original STREAM benchmark, it contains only the Copy kernel
 * while the specific kernel functions for different RD ratios are coded
 * in x86 assembly, using AVX instructions and non-temporal stores
 * (defined in utils.c file).
 * Also, the content of the arrays at the end is not checked.
 * We kept most of the comments from the original STREAM code.
 *
 * INSTRUCTIONS:
 *
 *	1) Benchmark requires different amounts of memory to run on different
 *     systems, depending on both the system cache size(s) and the
 *     granularity of the system timer.
 *     You should adjust the value of 'STREAM_ARRAY_SIZE' (below)
 *     to meet *both* of the following criteria:
 *       (a) Each array must be at least 4 times the size of the
 *           available cache memory. In practice, the minimum array size
 *           is about 3.8 times the cache size.
 *           Example 1: One Xeon E3 with 8 MB L3 cache
 *               STREAM_ARRAY_SIZE should be >= 4 million, giving
 *               an array size of 30.5 MB and a total memory requirement
 *               of 91.5 MB.
 *           Example 2: Two Xeon E5's with 20 MB L3 cache each (using OpenMP)
 *               STREAM_ARRAY_SIZE should be >= 20 million, giving
 *               an array size of 153 MB and a total memory requirement
 *               of 458 MB.
 *       (b) The size should be large enough so that the 'timing calibration'
 *           output by the program is at least 20 clock-ticks.
 *           Example: most versions of Windows have a 10 millisecond timer
 *               granularity.  20 "ticks" at 10 ms/tic is 200 milliseconds.
 *               If the chip is capable of 10 GB/s, it moves 2 GB in 200 msec.
 *               This means the each array must be at least 1 GB, or 128M elements.
 *
 *      Version 5.10 increases the default array size from 2 million
 *          elements to 10 million elements in response to the increasing
 *          size of L3 caches.  The new default size is large enough for caches
 *          up to 20 MB.
 *      Version 5.10 changes the loop index variables from "register int"
 *          to "ssize_t", which allows array indices >2^32 (4 billion)
 *          on properly configured 64-bit systems.  Additional compiler options
 *          (such as "-mcmodel=medium") may be required for large memory runs.
 *
 *      Array size can be set at compile time without modifying the source
 *          code for the (many) compilers that support preprocessor definitions
 *          on the compile line.  E.g.,
 *                icc -O -DSTREAM_ARRAY_SIZE=100000000 stream_mpi.c -o stream_mpi.100M
 *          will override the default size of 80M with a new size of 100M elements
 *          per array.
 */

/*
#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	400000000
#endif
*/

#ifdef NTIMES
#if NTIMES <= 0
#undef NTIMES
#define NTIMES 10
#endif
#endif
#ifndef NTIMES
#define NTIMES 10
#endif

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
    # define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
    # define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STREAM_TYPE
    #define STREAM_TYPE double
#endif

#define STREAM_KERNEL_GRAIN_ELEMS 400

// Some compilers require an extra keyword to recognize the "restrict" qualifier.
double * __restrict a, * __restrict b;
ssize_t array_elements, array_bytes, array_alignment;

const char *usage = "[-r <read_ratio>] [-p <pause>] [-s <array_size>] [-n <iterations>] [-P <period_ticks>] [-i] [-e] [-I] [-E]\n";

void (*STREAM_copy_rw)(double *a_array, double *b_array,
                         ssize_t *array_size, const int* const pause) = NULL;

int main(int argc, char *argv[])
{
    
    long long STREAM_ARRAY_SIZE = 0;	
    int BytesPerWord, k, rd_percentage = 50, opt;
    ssize_t j;
    int pause = 0;
    int run_iterations = NTIMES;
    long long periodic_stats_ticks = 0;
    int cli_skip_init = 0;
    int cli_skip_pre_m5_exit = 0;
    int cli_force_init = 0;
    int cli_force_pre_m5_exit = 0;

    // Command line parsing
    while (( opt = getopt(argc, argv, ":r:p:s:n:P:IEie")) != -1)
    {
        switch(opt)
        {
            case 'r':
                rd_percentage = atoi(optarg);
                if (rd_percentage < 0 || rd_percentage > 100 || rd_percentage % 2 == 1)
                {
                    printf("ERROR: RD ratio has to be even number between 50 and 100.\n");
                    exit(-1);
                }
                break;
            case 'p':
                pause = atoi(optarg);
                if (pause < 0)
                {
                    printf("ERROR: Pause has to be a non-negative number.\n");
                    exit(-1);
                }
                break;
            case 's':
                STREAM_ARRAY_SIZE = atoll(optarg);
                break;
            case 'n':
                run_iterations = atoi(optarg);
                if (run_iterations <= 0)
                {
                    printf("ERROR: Iterations must be a positive integer.\n");
                    exit(-1);
                }
                break;
            case 'P':
                periodic_stats_ticks = atoll(optarg);
                if (periodic_stats_ticks < 0)
                {
                    printf("ERROR: periodic stats ticks must be >= 0.\n");
                    exit(-1);
                }
                break;
            case 'I':
                cli_force_init = 1;
                break;
            case 'i':
                cli_skip_init = 1;
                break;
            case 'E':
                cli_force_pre_m5_exit = 1;
                break;
            case 'e':
                cli_skip_pre_m5_exit = 1;
                break;
	    default:
                print_usage(argv, (char *)usage);
                exit(-1);
        }
    }

    if (optind < argc || STREAM_ARRAY_SIZE == 0)
    {
        if (STREAM_ARRAY_SIZE == 0) printf("ERROR: array size must be > 0. Please specify -s <size>\n");
        print_usage(argv, (char *)usage);
        exit(-1);
    }
    {
        char data_json[256];
        snprintf(data_json, sizeof(data_json),
                 "{\"streamArraySize\":%lld,\"rdPercentage\":%d,\"pause\":%d,"
                 "\"optind\":%d,\"argc\":%d,\"cliSkipInit\":%d,\"cliSkipPreM5Exit\":%d,"
                 "\"cliForceInit\":%d,\"cliForcePreM5Exit\":%d,\"runIterations\":%d,"
                 "\"periodicStatsTicks\":%lld}",
                 STREAM_ARRAY_SIZE,
                 rd_percentage,
                 pause,
                 optind,
                 argc,
                 cli_skip_init,
                 cli_skip_pre_m5_exit,
                 cli_force_init,
                 cli_force_pre_m5_exit,
                 run_iterations,
                 periodic_stats_ticks);
        // #region agent log
        debug_log_json("pre-fix", "H1", "stream_omp.c:main:post-parse",
                       "Parsed CLI arguments", data_json);
        // #endregion
    }

    // End of command line partsing

    // Assigning the right asm function based on the RD ratio
    switch(rd_percentage)
    {
        case 0:
            STREAM_copy_rw = &STREAM_copy_0;
            break;
        case 2:
            STREAM_copy_rw = &STREAM_copy_2;
            break;
        case 4:
            STREAM_copy_rw = &STREAM_copy_4;
            break;
        case 6:
            STREAM_copy_rw = &STREAM_copy_6;
            break;
        case 8:
            STREAM_copy_rw = &STREAM_copy_8;
            break;
        case 10:
            STREAM_copy_rw = &STREAM_copy_10;
            break;
        case 12:
            STREAM_copy_rw = &STREAM_copy_12;
            break;
        case 14:
            STREAM_copy_rw = &STREAM_copy_14;
            break;
        case 16:
            STREAM_copy_rw = &STREAM_copy_16;
            break;
        case 18:
            STREAM_copy_rw = &STREAM_copy_18;
            break;
        case 20:
            STREAM_copy_rw = &STREAM_copy_20;
            break;
        case 22:
            STREAM_copy_rw = &STREAM_copy_22;
            break;
        case 24:
            STREAM_copy_rw = &STREAM_copy_24;
            break;
        case 26:
            STREAM_copy_rw = &STREAM_copy_26;
            break;
        case 28:
            STREAM_copy_rw = &STREAM_copy_28;
            break;
        case 30:
            STREAM_copy_rw = &STREAM_copy_30;
            break;
        case 32:
            STREAM_copy_rw = &STREAM_copy_32;
            break;
        case 34:
            STREAM_copy_rw = &STREAM_copy_34;
            break;
        case 36:
            STREAM_copy_rw = &STREAM_copy_36;
            break;
        case 38:
            STREAM_copy_rw = &STREAM_copy_38;
            break;
        case 40:
            STREAM_copy_rw = &STREAM_copy_40;
            break;
        case 42:
            STREAM_copy_rw = &STREAM_copy_42;
            break;
        case 44:
            STREAM_copy_rw = &STREAM_copy_44;
            break;
        case 46:
            STREAM_copy_rw = &STREAM_copy_46;
            break;
        case 48:
            STREAM_copy_rw = &STREAM_copy_48;
            break;
        case 50:
            STREAM_copy_rw = &STREAM_copy_50;
            break;
        case 52:
            STREAM_copy_rw = &STREAM_copy_52;
            break;
        case 54:
            STREAM_copy_rw = &STREAM_copy_54;
            break;
        case 56:
            STREAM_copy_rw = &STREAM_copy_56;
            break;
        case 58:
            STREAM_copy_rw = &STREAM_copy_58;
            break;
        case 60:
            STREAM_copy_rw = &STREAM_copy_60;
            break;
        case 62:
            STREAM_copy_rw = &STREAM_copy_62;
            break;
        case 64:
            STREAM_copy_rw = &STREAM_copy_64;
            break;
        case 66:
            STREAM_copy_rw = &STREAM_copy_66;
            break;
        case 68:
            STREAM_copy_rw = &STREAM_copy_68;
            break;
        case 70:
            STREAM_copy_rw = &STREAM_copy_70;
            break;
        case 72:
            STREAM_copy_rw = &STREAM_copy_72;
            break;
        case 74:
            STREAM_copy_rw = &STREAM_copy_74;
            break;
        case 76:
            STREAM_copy_rw = &STREAM_copy_76;
            break;
        case 78:
            STREAM_copy_rw = &STREAM_copy_78;
            break;
        case 80:
            STREAM_copy_rw = &STREAM_copy_80;
            break;
        case 82:
            STREAM_copy_rw = &STREAM_copy_82;
            break;
        case 84:
            STREAM_copy_rw = &STREAM_copy_84;
            break;
        case 86:
            STREAM_copy_rw = &STREAM_copy_86;
            break;
        case 88:
            STREAM_copy_rw = &STREAM_copy_88;
            break;
        case 90:
            STREAM_copy_rw = &STREAM_copy_90;
            break;
        case 92:
            STREAM_copy_rw = &STREAM_copy_92;
            break;
        case 94:
            STREAM_copy_rw = &STREAM_copy_94;
            break;
        case 96:
            STREAM_copy_rw = &STREAM_copy_96;
            break;
        case 98:
            STREAM_copy_rw = &STREAM_copy_98;
            break;
        case 100:
            STREAM_copy_rw = &STREAM_copy_100;
            break;
        default:
            STREAM_copy_rw = &STREAM_copy_50;
            break;
    }
    {
        char data_json[160];
        snprintf(data_json, sizeof(data_json),
                 "{\"rdPercentage\":%d,\"kernelPtr\":%llu}",
                 rd_percentage,
                 (unsigned long long)(uintptr_t)STREAM_copy_rw);
        // #region agent log
        debug_log_json("pre-fix", "H3", "stream_omp.c:main:kernel-select",
                       "Selected kernel function", data_json);
        // #endregion
    }


    /* --- distribute requested storage across MPI ranks --- */
    array_elements = STREAM_ARRAY_SIZE;              // don't worry about rounding vs truncation
    if (array_elements % STREAM_KERNEL_GRAIN_ELEMS != 0)
        array_elements += STREAM_KERNEL_GRAIN_ELEMS -
                          (array_elements % STREAM_KERNEL_GRAIN_ELEMS);
    array_alignment = 64;                                       // Can be modified -- provides partial support for adjusting relative alignment

    // Dynamically allocate the three arrays using "posix_memalign()"
    array_bytes = array_elements * sizeof(STREAM_TYPE);
    k = posix_memalign((void **)&a, array_alignment, array_bytes);
    if (k != 0)
    {
        printf("Allocation of array a failed, return code is %d\n",k);
        exit(1);
    }
    k = posix_memalign((void **)&b, array_alignment, array_bytes);
    if (k != 0)
    {
        printf("Allocation of array b failed, return code is %d\n",k);
        exit(1);
    }

    // Initial informational printouts -- rank 0 handles all the output
    if (1)
    {
        printf(HLINE);
        printf("$ Memory bandwidth load kernel $\n");
        printf(HLINE);
        BytesPerWord = sizeof(STREAM_TYPE);
        printf("This system uses %d bytes per array element.\n",
        BytesPerWord);

        printf(HLINE);
        #ifdef N
            printf("*****  WARNING: ******\n");
            printf("      It appears that you set the preprocessor variable N when compiling this code.\n");
            printf("      This version of the code uses the preprocesor variable STREAM_ARRAY_SIZE to control the array size\n");
            printf("      Reverting to default value of STREAM_ARRAY_SIZE=%llu\n",(unsigned long long) STREAM_ARRAY_SIZE);
            printf("*****  WARNING: ******\n");
        #endif

        printf("Total Aggregate Array size = %llu (elements)\n" , (unsigned long long) STREAM_ARRAY_SIZE);
        printf("Total Aggregate Memory per array = %.1f MiB (= %.1f GiB).\n",
          BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0),
          BytesPerWord * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.0/1024.0));
        printf("Total Aggregate memory required = %.1f MiB (= %.1f GiB).\n",
          (2.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024.),
          (2.0 * BytesPerWord) * ( (double) STREAM_ARRAY_SIZE / 1024.0/1024./1024.));

        printf(HLINE);
        printf("The kernel will be executed %d times.\n", run_iterations);

        #ifdef _OPENMP
            printf(HLINE);
            #pragma omp parallel
            {
                #pragma omp master
                {
                    k = omp_get_num_threads();
                    printf ("Number of Threads requested = %i\n",k);
                }
            }
        #endif

        #ifdef _OPENMP
            k = 0;
            #pragma omp parallel
            #pragma omp atomic
                k++;
                printf ("Number of Threads counted = %i\n",k);
        #endif

    }

    /* --- SETUP --- initialize arrays --- */
    {
        int skip_init = 1;
        if (cli_force_init)
            skip_init = 0;
        if (cli_skip_init)
            skip_init = 1;
        if (skip_init && rd_percentage >= 98)
        {
            /* H16 confirmed: skipping init with very high read ratios keeps
             * traffic off DRAM (zero-page effects). Force full init to reach
             * the expected high-bandwidth behavior for read-heavy kernels.
             */
            skip_init = 0;
            // #region agent log
            debug_log_json("pre-fix", "H16", "stream_omp.c:setup:auto-force-init",
                           "Overrode skip init for high read ratio",
                           "{\"reason\":\"high_read_ratio\",\"threshold\":98}");
            // #endregion
        }
        {
            char data_json[128];
            snprintf(data_json, sizeof(data_json),
                     "{\"skipInit\":%d,\"arrayElements\":%lld}",
                     skip_init, (long long)array_elements);
            // #region agent log
            debug_log_json("pre-fix", "H9", "stream_omp.c:setup:init-start",
                           "About to initialize arrays", data_json);
            // #endregion
        }
        if (!skip_init)
        {
#ifdef _OPENMP
            #pragma omp parallel for
#endif
            for (j=0; j<array_elements; j++)
            {
                a[j] = 1.0;
                b[j] = 2.0;
            }
        }
        {
            // #region agent log
            debug_log_json("pre-fix", "H9", "stream_omp.c:setup:init-end",
                           "Finished array initialization", "{\"ok\":1}");
            // #endregion
        }
    }

    /*	--- MAIN LOOP --- repeat the kernel like STREAM --- */
    
    {
        // #region agent log
        debug_log_json("pre-fix", "H6", "stream_omp.c:main:roi-enter",
                       "Entering ROI parallel section",
                       "{\"note\":\"before parallel region\"}");
        // #endregion
    }
    {
        int skip_pre_roi_exit = 1;
        if (cli_force_pre_m5_exit)
            skip_pre_roi_exit = 0;
        if (cli_skip_pre_m5_exit)
            skip_pre_roi_exit = 1;
        {
            char data_json[96];
            snprintf(data_json, sizeof(data_json),
                     "{\"skipPreM5Exit\":%d}", skip_pre_roi_exit);
            // #region agent log
            debug_log_json("pre-fix", "H8", "stream_omp.c:main:pre-m5-exit",
                           "About to execute pre-ROI m5_exit", data_json);
            // #endregion
        }
        if (!skip_pre_roi_exit)
        {
            // Trigger Python to switch from ATOMIC CPU to O3 CPU precisely before the ROI
            m5_exit(0);
            // #region agent log
            debug_log_json("pre-fix", "H8", "stream_omp.c:main:post-m5-exit",
                           "Returned from pre-ROI m5_exit", "{\"returned\":1}");
            // #endregion
        }
        else
        {
            // #region agent log
            debug_log_json("pre-fix", "H8", "stream_omp.c:main:post-m5-exit",
                           "Skipped pre-ROI m5_exit by env", "{\"returned\":0}");
            // #endregion
        }
    }

#ifdef _OPENMP
        #pragma omp parallel
#endif
    {
        int thread_id = 0;
        int thread_count = 1;
        int iter = 0;
        ssize_t total_blocks = array_elements / STREAM_KERNEL_GRAIN_ELEMS;
        ssize_t chunk = total_blocks;
        ssize_t remainder = 0;
        ssize_t local_blocks = total_blocks;
        ssize_t local_start = 0;
        ssize_t local_elements = array_elements;

#ifdef _OPENMP
        thread_id = omp_get_thread_num();
        thread_count = omp_get_num_threads();
#endif

        chunk = total_blocks / thread_count;
        remainder = total_blocks % thread_count;
        local_blocks = chunk + (thread_id < remainder ? 1 : 0);
        local_start = ((thread_id * chunk) + MIN(thread_id, remainder)) *
                      STREAM_KERNEL_GRAIN_ELEMS;
        local_elements = local_blocks * STREAM_KERNEL_GRAIN_ELEMS;
        if (thread_id < 4)
        {
            char data_json[320];
            snprintf(data_json, sizeof(data_json),
                     "{\"threadId\":%d,\"threadCount\":%d,\"totalBlocks\":%lld,"
                     "\"chunk\":%lld,\"remainder\":%lld,\"localBlocks\":%lld,"
                     "\"localStart\":%lld,\"localElements\":%lld,\"arrayElements\":%lld}",
                     thread_id,
                     thread_count,
                     (long long)total_blocks,
                     (long long)chunk,
                     (long long)remainder,
                     (long long)local_blocks,
                     (long long)local_start,
                     (long long)local_elements,
                     (long long)array_elements);
            // #region agent log
            debug_log_json("pre-fix", "H2", "stream_omp.c:parallel:partition",
                           "Computed thread partition", data_json);
            // #endregion
        }

#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        {
            // #region agent log
            debug_log_json("pre-fix", "H6", "stream_omp.c:parallel:reset",
                           "Issuing m5_dump_reset_stats", "{\"ok\":1}");
            // #endregion
            m5_dump_reset_stats(0, 0);
            if (periodic_stats_ticks > 0)
            {
                char data_json[128];
                snprintf(data_json, sizeof(data_json),
                         "{\"periodTicks\":%lld}", periodic_stats_ticks);
                // #region agent log
                debug_log_json("pre-fix", "H18", "stream_omp.c:parallel:periodic-stats",
                               "Enabled periodic m5_dump_stats", data_json);
                // #endregion
                m5_dump_stats(0, (uint64_t)periodic_stats_ticks);
            }
        }
#ifdef _OPENMP
        #pragma omp barrier
#endif

        for (iter = 0; iter < run_iterations; iter++)
        {
            if (thread_id == 0)
            {
                char data_json[96];
                snprintf(data_json, sizeof(data_json),
                         "{\"iter\":%d,\"runIterations\":%d}", iter, run_iterations);
                // #region agent log
                debug_log_json("pre-fix", "H11", "stream_omp.c:parallel:iter-start",
                               "Starting kernel iteration", data_json);
                // #endregion
            }
            if (local_elements > 0)
            {
                if (iter == 0 && thread_id == 0)
                {
                    // #region agent log
                    debug_log_json("pre-fix", "H7", "stream_omp.c:parallel:before-kernel",
                                   "About to call STREAM_copy_rw", "{\"threadId\":0,\"iter\":0}");
                    // #endregion
                }
                long long t_start = debug_now_ms();
                STREAM_copy_rw(a + local_start, b + local_start, &local_elements, &pause);
                if (iter == 0 && thread_id < 2)
                {
                    long long t_end = debug_now_ms();
                    char data_json[256];
                    snprintf(data_json, sizeof(data_json),
                             "{\"threadId\":%d,\"iter\":%d,\"localElements\":%lld,"
                             "\"elapsedMs\":%lld,\"pause\":%d}",
                             thread_id,
                             iter,
                             (long long)local_elements,
                             (long long)(t_end - t_start),
                             pause);
                    // #region agent log
                    debug_log_json("pre-fix", "H4", "stream_omp.c:parallel:kernel-call",
                                   "Kernel call timing sample", data_json);
                    // #endregion
                }
            }
            if (thread_id == 0)
            {
                char data_json[96];
                snprintf(data_json, sizeof(data_json),
                         "{\"iter\":%d,\"runIterations\":%d}", iter, run_iterations);
                // #region agent log
                debug_log_json("pre-fix", "H11", "stream_omp.c:parallel:iter-end",
                               "Finished kernel iteration", data_json);
                // #endregion
            }
        }

#ifdef _OPENMP
        if (thread_id < 4)
        {
            char data_json[160];
            snprintf(data_json, sizeof(data_json),
                     "{\"threadId\":%d,\"localElements\":%lld,\"runIterations\":%d}",
                     thread_id, (long long)local_elements, run_iterations);
            // #region agent log
            debug_log_json("pre-fix", "H14", "stream_omp.c:parallel:before-final-barrier",
                           "Thread reached final barrier point", data_json);
            // #endregion
        }
        #pragma omp barrier
        #pragma omp master
#endif
        {
            char data_json[160];
            snprintf(data_json, sizeof(data_json),
                     "{\"ntimes\":%d,\"arrayElements\":%lld}",
                     run_iterations,
                     (long long)array_elements);
            // #region agent log
            debug_log_json("pre-fix", "H5", "stream_omp.c:parallel:roi-end",
                           "Reached ROI end before dump_stats", data_json);
            // #endregion
            m5_dump_stats(0, 0);
            // End the simulation immediately after the timed region completes.
            m5_exit(0);
        }
    }

    free(a);
    free(b);

    return(0);
}

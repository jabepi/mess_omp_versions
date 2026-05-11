# define _XOPEN_SOURCE 600
# define STREAM_ARRAY_ALIGNMENT 64 // cache-line alignment

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
# include "utils.h"

static long long debug_now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((long long)tv.tv_sec * 1000LL) + ((long long)tv.tv_usec / 1000LL);
}
static void debug_log_json(const char *message)
{
    fprintf(stdout, "{\"message\":\"%s\",\"timestamp\":%lld}\n", message, debug_now_ms());
    fflush(stdout);
}

// Gem5 functions 
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

double * __restrict a, * __restrict b;
ssize_t array_elements, array_bytes, array_alignment;

const char *usage = "[-r <read_ratio>] [-p <pause>] [-s <array_size>] [-n <iterations>] [-P <period_ticks>] [-i] [-m] [-d] [-h]\n";

void (*STREAM_copy_rw)(double *a_array, double *b_array,
                         ssize_t *array_size, const int* const pause) = NULL;

typedef struct {
    long long stream_array_size;
    int       rd_percentage;
    int       pause_value;
    int       run_iterations;
    long long periodic_stats_ticks;
    int       cli_skip_init;
    int       m5_enabled;
    int       debug_enabled;
} cli_options;

static void parse_args(int argc, char *argv[], cli_options *opts)
{
    int opt;
    while ((opt = getopt(argc, argv, ":r:p:s:n:P:imdh")) != -1)
    {
        switch (opt)
        {
            case 'r':
                opts->rd_percentage = atoi(optarg);
                if (opts->rd_percentage < 0 || opts->rd_percentage > 100 || opts->rd_percentage % 2 != 0)
                {
                    printf("ERROR: RD ratio has to be even number between 0 and 100.\n");
                    exit(-1);
                }
                break;
            case 'p':
                opts->pause_value = atoi(optarg);
                if (opts->pause_value < 0)
                {
                    printf("ERROR: Pause has to be a non-negative number.\n");
                    exit(-1);
                }
                break;
            case 's':
                opts->stream_array_size = atoll(optarg);
                if (opts->stream_array_size <= 0)
                {
                    printf("ERROR: Array size must be > 0. Please specify -s <size>\n");
                    exit(-1);
                }
                break;
            case 'n':
                opts->run_iterations = atoi(optarg);
                if (opts->run_iterations <= 0)
                {
                    printf("ERROR: Iterations must be a positive integer.\n");
                    exit(-1);
                }
                break;
            case 'P':
                opts->periodic_stats_ticks = atoll(optarg);
                if (opts->periodic_stats_ticks < 0)
                {
                    printf("ERROR: periodic stats ticks must be >= 0.\n");
                    exit(-1);
                }
                break;
            case 'i':
                opts->cli_skip_init = 1;
                break;
            case 'm':
                opts->m5_enabled = 1;
                break;
            case 'd':
                opts->debug_enabled = 1;
                break;
            case 'h':
                printf("Usage: %s %s", argv[0], usage);
                printf("Options:\n");
                printf("  -r <read_ratio>         Set the read ratio (number between 0 and 100)\n");
                printf("  -p <pause>              Set pause duration (non-negative integer)\n");
                printf("  -s <array_size>         Set array size (positive integer)\n");
                printf("  -n <iterations>         Set number of kernel iterations (positive integer)\n");
                printf("  -P <period_ticks>       Set periodic statistics ticks interval (>= 0) waited to dump stats \n");
                printf("  -i                      Skip stream array initialization\n");
                printf("  -m                      Enable gem5 m5_* calls for gem5 (m5_exit, m5_dump_stats, ...)\n");
                printf("  -d                      Enable debug logging\n");
                printf("  -h                      Show this help message\n");
                exit(0);
            default:
                print_usage(argv, (char *)usage);
                exit(-1);
        }
    }
}

int main(int argc, char *argv[])
{
    
    int BytesPerWord, k;
    ssize_t j;

    cli_options opts = {
        .stream_array_size      = 0,
        .rd_percentage          = 100,
        .pause_value            = 0,
        .run_iterations         = 1,
        .periodic_stats_ticks   = 10000,
        .cli_skip_init          = 0,
        .m5_enabled             = 0,
        .debug_enabled          = 0,
    };
    parse_args(argc, argv, &opts);

    long long STREAM_ARRAY_SIZE     = opts.stream_array_size;
    int       rd_percentage         = opts.rd_percentage;
    int       pause                 = opts.pause_value;
    int       run_iterations        = opts.run_iterations;
    long long periodic_stats_ticks  = opts.periodic_stats_ticks;
    int       cli_skip_init         = opts.cli_skip_init;
    int       m5_enabled            = opts.m5_enabled;
    int       debug_enabled         = opts.debug_enabled;

    if (debug_enabled)
        {
            char dbg_msg[512];
            snprintf(dbg_msg, sizeof(dbg_msg),
                "Command line arguments: rd_percentage=%d, pause=%d, array_size=%lld, iterations=%d, periodic_stats_ticks=%lld, skip_init=%d, m5_enabled=%d, debug_enabled=%d",
                rd_percentage, pause, STREAM_ARRAY_SIZE, run_iterations,
                periodic_stats_ticks, cli_skip_init, m5_enabled, debug_enabled);
            debug_log_json(dbg_msg);
        }
   
    
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

    /* --- distribute requested storage across OpenMP threads --- */
    
    // Round up the array size to the nearest multiple of the kernel grain size
    array_elements = STREAM_ARRAY_SIZE;             
    if (array_elements % STREAM_KERNEL_GRAIN_ELEMS != 0)
        array_elements += STREAM_KERNEL_GRAIN_ELEMS -
                          (array_elements % STREAM_KERNEL_GRAIN_ELEMS);
    array_alignment = STREAM_ARRAY_ALIGNMENT; 

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
    if (debug_enabled)
    {
        printf(HLINE);
        printf("$ Memory bandwidth load kernel $\n");
        printf(HLINE);
        BytesPerWord = sizeof(STREAM_TYPE);
        printf("This system uses %d bytes per array element.\n",
        BytesPerWord);

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
    
    if (!cli_skip_init)
    {
        if (debug_enabled)
            debug_log_json("Starting array initialization setup");

        for (j=0; j<array_elements; j++)
        {
            a[j] = 1.0;
            b[j] = 2.0;
        }
        if (debug_enabled)
            debug_log_json("Finished array initialization setup");
    }
    
    /*	--- MAIN LOOP --- repeat the kernel like STREAM --- */
    if (debug_enabled)
        debug_log_json("Entering ROI parallel section");
    

#ifdef _OPENMP
        #pragma omp parallel
#endif
    {
        int thread_id = 0;
        int thread_count = 1;
        int iter;
        // total_blocks: total number of STREAM_KERNEL_GRAIN_ELEMS-sized blocks
        // that make up the whole working set (array_elements is rounded up to
        // a multiple of the grain, so this division is exact).
        ssize_t total_blocks = array_elements / STREAM_KERNEL_GRAIN_ELEMS;
        // chunk: base number of blocks each thread gets when total_blocks is
        // divided as evenly as possible across thread_count threads.
        ssize_t chunk = total_blocks;
        // remainder: blocks left over after the even split; the first
        // `remainder` threads each receive one extra block.
        ssize_t remainder = 0;
        // local_blocks: number of blocks this particular thread will process
        // (chunk, plus one extra if this thread is among the first `remainder`).
        ssize_t local_blocks = total_blocks;
        // local_start: starting element index (offset into a/b) for this
        // thread's slice; always a multiple of STREAM_KERNEL_GRAIN_ELEMS so
        // each thread starts on a kernel-block boundary.
        ssize_t local_start = 0;
        // local_elements: length, in elements, of this thread's slice
        // (= local_blocks * STREAM_KERNEL_GRAIN_ELEMS); passed to the kernel.
        ssize_t local_elements = array_elements;

#ifdef _OPENMP
        thread_id = omp_get_thread_num();
        thread_count = omp_get_num_threads();
#endif

        // Per-thread partitioning of the total_blocks across thread_count
        // OpenMP threads, distributing any remainder one block at a time to
        // the lowest-numbered threads.
        chunk = total_blocks / thread_count;
        remainder = total_blocks % thread_count;
        local_blocks = chunk + (thread_id < remainder ? 1 : 0);
        local_start = ((thread_id * chunk) + MIN(thread_id, remainder)) *
                      STREAM_KERNEL_GRAIN_ELEMS;
        local_elements = local_blocks * STREAM_KERNEL_GRAIN_ELEMS;
        if (debug_enabled && thread_id < 4)
            debug_log_json("Computed thread partition for STREAM kernel");

#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        {
            if (m5_enabled)
            {
                if (debug_enabled)
                    debug_log_json("Resetting gem5 statistics before timed region");
                m5_dump_reset_stats(0, 0);

                if (debug_enabled)
                    debug_log_json("Enabling periodic gem5 statistics dumps");
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
                if (debug_enabled)
                    debug_log_json("Starting STREAM kernel iteration");
            }
            if (local_elements > 0)
            {
                if (iter == 0 && thread_id == 0)
                {
                    if (debug_enabled)
                        debug_log_json("Calling STREAM kernel function for the first time");
                }
                STREAM_copy_rw(a + local_start, b + local_start, &local_elements, &pause);
                if (debug_enabled && iter == 0 && thread_id < 2)
                    debug_log_json("Finished first STREAM kernel call sample");
            }
            if (thread_id == 0)
            {
                if (debug_enabled)
                    debug_log_json("Finished STREAM kernel iteration");
            }
        }

#ifdef _OPENMP
        #pragma omp barrier
        if (debug_enabled && thread_id < 4)
            debug_log_json("Thread reached final ROI barrier");
        #pragma omp master
#endif
        {
            if (m5_enabled)
            {
                if (debug_enabled)
                    debug_log_json("Leaving ROI and dumping final gem5 statistics");
                m5_dump_stats(0, 0);
            }
        }
    }
    if (m5_enabled)
        m5_exit(0);
    
    free(a);
    free(b);
    return(0);
}

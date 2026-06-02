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
# include <errno.h>
# include <time.h>
# include <omp.h>
# include "utils.h"
# include "aux.h"


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


const char *usage = "[-r <read_ratio>] [-p <pause>] [-s <array_size>] [-n <iterations>] "
                    "[-P <period_ticks>] [-c <chase_elems>] [-x <chase_total_loads>] "
                    "[-w <walk_file>] [-t <0|1>] [-u <warmup_iters>] [-i] [-m] [-d] [-h]\n";

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
    long long chase_array_elems;
    uint64_t  chase_total_loads;
    const char *walk_file_path;
    int       thread0_pointer_chase;
    int       warmup_iterations;
} cli_options;

static void parse_args(int argc, char *argv[], cli_options *opts)
{
    int opt;
    while ((opt = getopt(argc, argv, ":r:p:s:n:P:c:x:w:t:u:imdh")) != -1)
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
            case 'c':
                opts->chase_array_elems = atoll(optarg);
                if (opts->chase_array_elems <= 0)
                {
                    printf("ERROR: pointer-chase elements must be > 0.\n");
                    exit(-1);
                }
                break;
            case 'x':
                opts->chase_total_loads = (uint64_t)strtoull(optarg, NULL, 10);
                if (opts->chase_total_loads == 0ULL)
                {
                    printf("ERROR: pointer-chase total loads must be > 0.\n");
                    exit(-1);
                }
                break;
            case 'w':
                opts->walk_file_path = optarg;
                if (opts->walk_file_path[0] == '\0')
                {
                    printf("ERROR: walk file path cannot be empty.\n");
                    exit(-1);
                }
                break;
            case 't':
                opts->thread0_pointer_chase = atoi(optarg);
                if (opts->thread0_pointer_chase != 0 && opts->thread0_pointer_chase != 1)
                {
                    printf("ERROR: thread0 pointer-chase selector must be 0 or 1.\n");
                    exit(-1);
                }
                break;
            case 'u':
                opts->warmup_iterations = atoi(optarg);
                if (opts->warmup_iterations < 0)
                {
                    printf("ERROR: warmup iterations must be >= 0.\n");
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
                printf("  -c <chase_elems>        Set pointer-chase nodes (cache-line nodes, > 0)\n");
                printf("  -x <chase_total_loads>  Set pointer-chase total dependent loads per kernel call (> 0)\n");
                printf("  -w <walk_file>          Pointer-walk file path to load/save\n");
                printf("  -t <0|1>                Enable thread 0 pointer-chase role (1 enabled)\n");
                printf("  -u <warmup_iters>       STREAM-only warmup iterations before pointer-chase measurement\n");
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

    // Post-parse defaults/cross-option checks (not duplicated in per-case validation).
    if (opts->chase_array_elems == 0)
        opts->chase_array_elems = (long long)(POINTER_CHASE_DEFAULT_BYTES / POINTER_CHASE_CACHE_LINE);
}

int main(int argc, char *argv[])
{
    
    int k;
    ssize_t j;

    cli_options opts = {
        .stream_array_size      = 0,
        .rd_percentage          = 100,
        .pause_value            = 0,
        .run_iterations         = 12,
        .periodic_stats_ticks   = 10000,
        .cli_skip_init          = 0,
        .m5_enabled             = 0,
        .debug_enabled          = 0,
        .chase_array_elems      = 0,
        .chase_total_loads      = 320000ULL,
        .walk_file_path         = "array.dat",
        .thread0_pointer_chase  = 0,
        .warmup_iterations      = 4,
    };
    parse_args(argc, argv, &opts);


    // Auxiliary variables
    ssize_t chase_array_bytes       = 0;
    struct pointer_chase_line *chase_array = NULL;
    volatile uint64_t chase_sink = 0;
    uint64_t pointer_chase_total_cycles = 0;
    unsigned long long pointer_chase_total_loads = 0ULL;
    uint64_t pointer_chase_next_offset = 0;
    volatile int stream_workers_stop = 0;
    volatile int start_pointer_chase = 0;
    uint64_t arch_timer_hz = cycles_per_second();

    if (opts.debug_enabled)
        {
            char dbg_msg[512];
            snprintf(dbg_msg, sizeof(dbg_msg),
                "Command line arguments: rd_percentage=%d, pause=%d, array_size=%lld, iterations=%d, periodic_stats_ticks=%lld, skip_init=%d, m5_enabled=%d, debug_enabled=%d, chase_nodes=%lld, chase_total_loads=%llu, walk_file=%s, activate_ptchase=%d, warmup_iterations=%d",
                opts.rd_percentage, opts.pause_value, opts.stream_array_size, opts.run_iterations,
                opts.periodic_stats_ticks, opts.cli_skip_init, opts.m5_enabled, opts.debug_enabled,
                opts.chase_array_elems, (unsigned long long)opts.chase_total_loads, opts.walk_file_path, opts.thread0_pointer_chase, opts.warmup_iterations);
            debug_log_json(dbg_msg);
        }

    // Assigning the right asm function based on the RD ratio
    switch(opts.rd_percentage)
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

    // ------------------------------------------------------------
    // ALLOCATE ARRAYS
    // ------------------------------------------------------------
    
    // 1. Allocate STREAM arrays
    if (opts.stream_array_size % STREAM_KERNEL_GRAIN_ELEMS != 0)
        opts.stream_array_size += STREAM_KERNEL_GRAIN_ELEMS -
                                  (opts.stream_array_size % STREAM_KERNEL_GRAIN_ELEMS);
    ssize_t array_bytes = (ssize_t)opts.stream_array_size * (ssize_t)sizeof(STREAM_TYPE);
    k = posix_memalign((void **)&a, STREAM_ARRAY_ALIGNMENT, array_bytes);
    if (k != 0)
    {
        printf("Allocation of array a failed, return code is %d\n",k);
        exit(1);
    }
    k = posix_memalign((void **)&b, STREAM_ARRAY_ALIGNMENT, array_bytes);
    if (k != 0)
    {
        printf("Allocation of array b failed, return code is %d\n",k);
        exit(1);
    }

    // 2. Allocate and initialize pointer-chase array only when enabled.
    if (opts.thread0_pointer_chase)
    {
        chase_array_bytes = (ssize_t)opts.chase_array_elems * (ssize_t)sizeof(struct pointer_chase_line);
        k = posix_memalign((void **)&chase_array, POINTER_CHASE_CACHE_LINE, (size_t)chase_array_bytes);
        if (k != 0)
        {
            printf("Allocation of pointer-chase array failed, return code is %d\n",k);
            exit(1);
        }
        init_pointer_walk(opts.walk_file_path, chase_array, (uint64_t)opts.chase_array_elems);
    }

    // ------------------------------------------------------------
    //  DEBUG PRINTOUTS
    // ------------------------------------------------------------
    if (opts.debug_enabled)
    {
        print_initial_kernel_info((int)sizeof(STREAM_TYPE),
                                  opts.stream_array_size,
                                  opts.chase_array_elems,
                                  chase_array_bytes,
                                  opts.thread0_pointer_chase,
                                  opts.chase_total_loads,
                                  opts.walk_file_path,
                                  opts.run_iterations);
    }

    // ------------------------------------------------------------
    //  INITIALIZE STREAM ARRAYS
    // ------------------------------------------------------------
    
    if (!opts.cli_skip_init)
    {
        if (opts.debug_enabled)
            debug_log_json("Starting array initialization setup");
        for (j = 0; j < (ssize_t)opts.stream_array_size; j++)
        {
            a[j] = 1.0;
            b[j] = 2.0;
        }
        if (opts.debug_enabled)
            debug_log_json("Finished array initialization setup");
    }
    
    // ------------------------------------------------------------
    //  MAIN LOOP
    // ------------------------------------------------------------
    if (opts.debug_enabled)
        debug_log_json("Entering main kernel section");
    

#ifdef _OPENMP
        #pragma omp parallel
#endif
    {   
        // 1. Split the stream arrays among the threads 
        // (each thread will process a contiguous chunk of the stream array)
        // ------------------------------------------------------------
        int thread_id = 0;
        int thread_count = 1;
        int stream_worker_count = 0;  // number of threads that will run the STREAM kernel
        int stream_worker_idx = -1;   // index of the current thread in the stream worker pool
        int iter;
        int warmup_iters = opts.warmup_iterations;
        
        // total_blocks: number of subdivions of the stream array
        ssize_t total_blocks = (ssize_t)opts.stream_array_size / STREAM_KERNEL_GRAIN_ELEMS;
        
        // chunk: number of blocks each thread will process
        ssize_t chunk = total_blocks;
        
        // remainder: number of blocks left over after the even split
        ssize_t remainder = 0;
        
        // local_blocks: number of blocks this particular thread will process
        ssize_t local_blocks = total_blocks;
        
        // local_start: starting element index (offset into a/b) for this thread's slice
        ssize_t local_start = 0;
        
        // local_elements: number of elements this thread will process
        ssize_t local_elements = (ssize_t)opts.stream_array_size;

#ifdef _OPENMP
        thread_id = omp_get_thread_num();
        thread_count = omp_get_num_threads();
#endif

        // Number of threads that will run the STREAM kernel
        // 1. If the pointer-chase is enabled, the number of threads is the total number of threads minus 1
        // 2. If the pointer-chase is disabled, the number of threads is the total number of threads
        stream_worker_count = opts.thread0_pointer_chase ?
                              (thread_count > 1 ? thread_count - 1 : 0) :
                              thread_count;

        
        // Note: stream worker count can be 0 if the pointer-chase is enabled and there is only one thread
        if (stream_worker_count > 0)
        {
            if (!opts.thread0_pointer_chase)
                stream_worker_idx = thread_id;
            else if (thread_id > 0)
                stream_worker_idx = thread_id - 1;
        }

        // Divide the stream array among the threads
        if (stream_worker_count > 0 && stream_worker_idx >= 0)
        {
            chunk = total_blocks / stream_worker_count;
            remainder = total_blocks % stream_worker_count;
            local_blocks = chunk + (stream_worker_idx < remainder ? 1 : 0);
            local_start = ((stream_worker_idx * chunk) + MIN(stream_worker_idx, remainder)) *
                          STREAM_KERNEL_GRAIN_ELEMS;
            local_elements = local_blocks * STREAM_KERNEL_GRAIN_ELEMS;
        }
        else
        {
            local_blocks = 0;
            local_start = 0;
            local_elements = 0;
        }

        if (opts.debug_enabled && thread_id < 4)
            debug_log_json("Computed thread partition for STREAM kernel");
        // ------------------------------------------------------------

#ifdef _OPENMP
        #pragma omp barrier
        #pragma omp master
#endif
        {

        // 2. Reset gem5 statistics if enabled
            if (opts.m5_enabled)
            {
                if (opts.debug_enabled)
                    debug_log_json("Resetting gem5 statistics before timed region");
                m5_dump_reset_stats(0, 0);

                if (opts.debug_enabled)
                    debug_log_json("Enabling periodic gem5 statistics dumps");
                m5_dump_stats(0, (uint64_t)opts.periodic_stats_ticks);
            }
        }
#ifdef _OPENMP
        #pragma omp barrier
#endif


        // ------------------------------------------------------------
        //  IF POINTER-CHASE IS ENABLED
        // ------------------------------------------------------------
        if (opts.thread0_pointer_chase)
        {
            if (thread_id == 0)
            {
                stream_workers_stop = 0;
                if (stream_worker_count == 0)
                    start_pointer_chase = 1;
#ifdef _OPENMP
                #pragma omp flush(stream_workers_stop, start_pointer_chase)
#endif
            }

            // ------------------------------------------------------------
            //  IF THIS IS A STREAM WORKER
            // ------------------------------------------------------------
            if (thread_id > 0 && local_elements > 0)
            {
                // 1. STREAM workers run warmup first.
                for (iter = 0; iter < warmup_iters; iter++)
                    STREAM_copy_rw(a + local_start, b + local_start, &local_elements, &opts.pause_value);

                // Notify thread 0 that overlap can start.
                if (stream_worker_idx == 0)
                {
                    start_pointer_chase = 1;
#ifdef _OPENMP
                    #pragma omp flush(start_pointer_chase)
#endif
                }

                // After warmup, keep running STREAM until thread 0 signals stop.
                while (1)
                {
#ifdef _OPENMP
                        #pragma omp flush(stream_workers_stop)
#endif
                        if (stream_workers_stop)
                            break;
                        STREAM_copy_rw(a + local_start, b + local_start, &local_elements, &opts.pause_value);
                }
                
            }
            
            // ------------------------------------------------------------
            //  IF THIS IS THE POINTER-CHASE THREAD
            // ------------------------------------------------------------
            if (thread_id == 0)
            {
#ifdef _OPENMP
                while (!start_pointer_chase)
                {
                    #pragma omp flush(start_pointer_chase)
                }
#endif
                // Run the pointer-chase kernel 
                for (iter = 0; iter < opts.run_iterations; iter++)
                {
                    uint64_t kernel_cycles = 0;
                    uint64_t chase_value = pointer_chase_kernel(chase_array,
                                                                (uint64_t)opts.chase_array_elems,
                                                                opts.chase_total_loads,
                                                                &pointer_chase_next_offset,
                                                                &kernel_cycles);
                    chase_sink ^= chase_value; //TO PREVENT COMPILER OPTIMIZATION
                    pointer_chase_total_cycles += kernel_cycles;
                    pointer_chase_total_loads += (unsigned long long)opts.chase_total_loads;
                }
                stream_workers_stop = 1;
#ifdef _OPENMP
                #pragma omp flush(stream_workers_stop)
#endif
                // Calculate the latency of the pointer-chase kernel
                double latency_sim_ns = 0.0;
                double latency_cycles = 0.0;
                double total_latency_ns = 0.0;
                if (pointer_chase_total_loads > 0ULL && pointer_chase_total_cycles > 0ULL)
                    latency_cycles = (double)pointer_chase_total_cycles /
                                        (double)pointer_chase_total_loads;
                if (pointer_chase_total_cycles > 0ULL && arch_timer_hz > 0ULL)
                    total_latency_ns = (double)pointer_chase_total_cycles * (1.0e9 / (double)arch_timer_hz);
                if (latency_cycles > 0.0 && arch_timer_hz > 0ULL)
                    latency_sim_ns = latency_cycles * (1.0e9 / (double)arch_timer_hz);
                printf("Pointer-chase latency: total_ns=%.6f avg_ns_per_access=%.6f\n",
                        total_latency_ns, latency_sim_ns);
                
            }
        }
        else
        {
        // ------------------------------------------------------------
        //  IF THE POINTER-CHASE IS DISABLED
        // ------------------------------------------------------------
            for (iter = 0; iter < opts.run_iterations; iter++)
            {
                if (thread_id == 0)
                {
                    if (opts.debug_enabled)
                        debug_log_json("Starting STREAM kernel iteration");
                }
                if (local_elements > 0)
                {
                    if (iter == 0 && thread_id == 0)
                    {
                        if (opts.debug_enabled)
                            debug_log_json("Calling STREAM kernel function for the first time");
                    }
                    STREAM_copy_rw(a + local_start, b + local_start, &local_elements, &opts.pause_value);
                    if (opts.debug_enabled && iter == 0 && thread_id < 2)
                        debug_log_json("Finished first STREAM kernel call sample");
                }
                if (thread_id == 0)
                {
                    if (opts.debug_enabled)
                        debug_log_json("Finished STREAM kernel iteration");
                }
            }
        }

#ifdef _OPENMP
        #pragma omp barrier
        if (opts.debug_enabled && thread_id < 4)
            debug_log_json("Thread reached final ROI barrier");
        #pragma omp master
#endif
        {
            if (opts.m5_enabled)
            {
                if (opts.debug_enabled)
                    debug_log_json("Leaving ROI and dumping final gem5 statistics");
                m5_dump_stats(0, 0);
            }
        }
    }
    if (opts.m5_enabled)
        m5_exit(0);
    
    free(a);
    free(b);
    free(chase_array);
    return(0);
}

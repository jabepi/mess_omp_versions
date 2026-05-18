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

static long long debug_now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((long long)tv.tv_sec * 1000LL) + ((long long)tv.tv_usec / 1000LL);
}
static uint64_t debug_now_ms_u64(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t)ts.tv_sec * 1000ULL) + (uint64_t)(ts.tv_nsec / 1000000ULL);
}
static void debug_emit_stdout(const char *run_id,
                              const char *hypothesis_id,
                              const char *location,
                              const char *message,
                              const char *data_json)
{
    fprintf(stdout,
            "{\"sessionId\":\"71e020\",\"runId\":\"%s\",\"hypothesisId\":\"%s\","
            "\"location\":\"%s\",\"message\":\"%s\",\"data\":%s,\"timestamp\":%llu}\n",
            run_id,
            hypothesis_id,
            location,
            message,
            data_json ? data_json : "{}",
            (unsigned long long)debug_now_ms_u64());
    fflush(stdout);
}
static void debug_log_json(const char *message)
{
    fprintf(stdout, "{\"message\":\"%s\",\"timestamp\":%lld}\n", message, debug_now_ms());
    fflush(stdout);
}
static uint64_t now_cycles(void)
{
#if defined(__aarch64__)
    uint64_t cyc = 0;
    __asm__ __volatile__(
        "isb\n\t"
        "mrs %0, cntvct_el0\n\t"
        : "=r"(cyc));
    return cyc;
#else
    return 0;
#endif
}
static uint64_t cycles_per_second(void)
{
#if defined(__aarch64__)
    uint64_t freq = 0;
    __asm__ __volatile__(
        "mrs %0, cntfrq_el0\n\t"
        : "=r"(freq));
    return freq;
#else
    return 0;
#endif
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
#define POINTER_CHASE_CACHE_LINE 128
#define POINTER_CHASE_DEFAULT_BYTES (100ULL * 1024ULL * 1024ULL)

double * __restrict a, * __restrict b;
ssize_t array_elements, array_bytes, array_alignment;

struct pointer_chase_line
{
    uint64_t next_offset;
    uint8_t pad[POINTER_CHASE_CACHE_LINE - sizeof(uint64_t)];
};

static void shuffle_u64(uint64_t *array, uint64_t n)
{
    uint64_t i;

    srand(0);
    if (n <= 1)
        return;

    for (i = 0; i < n - 1; i++)
    {
        uint64_t j = i + (uint64_t)(rand() / (RAND_MAX / (n - i) + 1));
        uint64_t t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

static void generate_pointer_walk(struct pointer_chase_line *walk_array, uint64_t elems)
{
    uint64_t *perm;
    uint64_t j;

    if (elems == 0)
        return;

    if (elems == 1)
    {
        walk_array[0].next_offset = 0;
        return;
    }

    perm = (uint64_t *)malloc(elems * sizeof(uint64_t));

    if (perm == NULL)
    {
        fprintf(stderr, "WARNING: pointer-chase permutation allocation failed; using sequential ring.\n");
        for (j = 0; j < elems; j++)
            walk_array[j].next_offset = ((j + 1) % elems) * POINTER_CHASE_CACHE_LINE;
        free(perm);
        return;
    }

    for (j = 0; j < elems; j++)
        perm[j] = j;

    shuffle_u64(perm, elems);

    for (j = 0; j < elems; j++)
    {
        uint64_t cur = perm[j];
        uint64_t next = perm[(j + 1) % elems];
        walk_array[cur].next_offset = next * POINTER_CHASE_CACHE_LINE;
    }

    free(perm);
}

static int validate_pointer_walk(const struct pointer_chase_line *walk_array, uint64_t elems)
{
    uint8_t *in_degree = NULL;
    uint8_t *visited = NULL;
    uint64_t i;
    uint64_t node = 0;
    int ok = 0;

    if (walk_array == NULL || elems == 0)
        return 0;

    in_degree = (uint8_t *)calloc(elems, sizeof(uint8_t));
    visited = (uint8_t *)calloc(elems, sizeof(uint8_t));
    if (in_degree == NULL || visited == NULL)
        goto cleanup;

    for (i = 0; i < elems; i++)
    {
        uint64_t next_offset = walk_array[i].next_offset;
        uint64_t next_idx;
        if ((next_offset % POINTER_CHASE_CACHE_LINE) != 0)
            goto cleanup;
        next_idx = next_offset / POINTER_CHASE_CACHE_LINE;
        if (next_idx >= elems)
            goto cleanup;
        in_degree[next_idx]++;
        if (in_degree[next_idx] > 1)
            goto cleanup;
    }

    for (i = 0; i < elems; i++)
    {
        if (visited[node] != 0)
            goto cleanup;
        visited[node] = 1;
        node = walk_array[node].next_offset / POINTER_CHASE_CACHE_LINE;
    }

    if (node != 0)
        goto cleanup;
    for (i = 0; i < elems; i++)
    {
        if (visited[i] == 0)
            goto cleanup;
    }

    ok = 1;

cleanup:
    free(in_degree);
    free(visited);
    return ok;
}

static int load_pointer_walk_file(const char *walk_file_path,
                                  struct pointer_chase_line *walk_array,
                                  uint64_t elems)
{
    FILE *input_file;
    uint64_t i;
    unsigned long long tmp;
    uint64_t max_offset;

    if (walk_file_path == NULL || walk_file_path[0] == '\0')
        return -1;

    input_file = fopen(walk_file_path, "r");
    if (input_file == NULL)
        return -1;

    max_offset = elems * POINTER_CHASE_CACHE_LINE;
    for (i = 0; i < elems; i++)
    {
        if (fscanf(input_file, "%llu", &tmp) != 1)
        {
            fclose(input_file);
            return -1;
        }
        if ((tmp % POINTER_CHASE_CACHE_LINE) != 0 || tmp >= max_offset)
        {
            fclose(input_file);
            return -1;
        }
        walk_array[i].next_offset = (uint64_t)tmp;
    }

    fclose(input_file);
    if (!validate_pointer_walk(walk_array, elems))
        return -1;
    return 0;
}

static int save_pointer_walk_file(const char *walk_file_path,
                                  const struct pointer_chase_line *walk_array,
                                  uint64_t elems)
{
    FILE *output_file;
    uint64_t i;

    if (walk_file_path == NULL || walk_file_path[0] == '\0')
        return -1;

    output_file = fopen(walk_file_path, "w");
    if (output_file == NULL)
        return -1;

    for (i = 0; i < elems; i++)
    {
        if (fprintf(output_file, "%llu\n",
                    (unsigned long long)walk_array[i].next_offset) < 0)
        {
            fclose(output_file);
            return -1;
        }
    }

    if (fclose(output_file) != 0)
        return -1;

    return 0;
}

static void init_pointer_walk(const char *walk_file_path,
                              struct pointer_chase_line *walk_array,
                              uint64_t elems)
{
    if (load_pointer_walk_file(walk_file_path, walk_array, elems) == 0)
    {
        printf("Pointer walk loaded from '%s'.\n", walk_file_path);
        return;
    }

    printf("Pointer walk file '%s' unavailable or invalid; generating deterministic walk in-memory.\n",
           walk_file_path ? walk_file_path : "(null)");
    generate_pointer_walk(walk_array, elems);
    if (save_pointer_walk_file(walk_file_path, walk_array, elems) == 0)
    {
        printf("Pointer walk saved to '%s' for future executions.\n", walk_file_path);
    }
    else
    {
        printf("WARNING: failed to save pointer walk to '%s' (%s).\n",
               walk_file_path ? walk_file_path : "(null)",
               strerror(errno));
    }
}

static uint64_t pointer_chase_kernel(struct pointer_chase_line *walk_array,
                                     uint64_t elems,
                                     int chase_iterations,
                                     int chase_loads_per_iter,
                                     uint64_t *next_offset_state,
                                     uint64_t *kernel_cycles_out)
{
    uint64_t next_offset = 0;
    uint64_t total_loads = 0;
    uint64_t base_addr_u64;
    uint64_t max_offset;

    if (walk_array == NULL || elems == 0 || chase_iterations <= 0 || chase_loads_per_iter <= 0 ||
        next_offset_state == NULL)
        return 0;

    max_offset = elems * POINTER_CHASE_CACHE_LINE;
    if (*next_offset_state < max_offset && ((*next_offset_state % POINTER_CHASE_CACHE_LINE) == 0))
        next_offset = *next_offset_state;

    total_loads = (uint64_t)chase_iterations * (uint64_t)chase_loads_per_iter;
    base_addr_u64 = (uint64_t)(uintptr_t)walk_array;
#if defined(__aarch64__)
    {
        uint64_t remaining = 0;
        uint64_t next = 0;
        uint64_t base = 0;
        uint64_t begin_cycles = now_cycles();
        remaining = total_loads;
        next = next_offset;
        base = base_addr_u64;
        asm volatile(
            "cmp %0, #0\n\t"
            "beq 2f\n\t"
            "1:\n\t"
            "add x3, %2, %1\n\t"
            "ldr %1, [x3]\n\t"
            "subs %0, %0, #1\n\t"
            "bne 1b\n\t"
            "2:\n\t"
            : "+r"(remaining), "+r"(next)
            : "r"(base)
            : "x3", "cc", "memory");
        next_offset = next;
        if (kernel_cycles_out != NULL)
            *kernel_cycles_out += (now_cycles() - begin_cycles);
    }
#else
    {
        uint64_t step;
        uint8_t *base_addr = (uint8_t *)walk_array;
        uint64_t begin_cycles = now_cycles();
        for (step = 0; step < total_loads; step++)
        {
            volatile uint64_t *entry = (volatile uint64_t *)(base_addr + next_offset);
            next_offset = *entry;
        }
        if (kernel_cycles_out != NULL)
            *kernel_cycles_out += (now_cycles() - begin_cycles);
    }
#endif

    *next_offset_state = next_offset;
    return next_offset;
}

const char *usage = "[-r <read_ratio>] [-p <pause>] [-s <array_size>] [-n <iterations>] "
                    "[-P <period_ticks>] [-c <chase_elems>] [-x <chase_iters>] "
                    "[-l <chase_loads_per_iter>] [-w <walk_file>] [-t <0|1>] [-u <warmup_iters>] [-i] [-m] [-d] [-h]\n";

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
    int       chase_iterations;
    int       chase_loads_per_iter;
    const char *walk_file_path;
    int       thread0_pointer_chase;
    int       warmup_iterations;
} cli_options;

static void parse_args(int argc, char *argv[], cli_options *opts)
{
    int opt;
    while ((opt = getopt(argc, argv, ":r:p:s:n:P:c:x:l:w:t:u:imdh")) != -1)
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
                opts->chase_iterations = atoi(optarg);
                if (opts->chase_iterations <= 0)
                {
                    printf("ERROR: pointer-chase iterations must be > 0.\n");
                    exit(-1);
                }
                break;
            case 'l':
                opts->chase_loads_per_iter = atoi(optarg);
                if (opts->chase_loads_per_iter <= 0)
                {
                    printf("ERROR: pointer-chase loads per iter must be > 0.\n");
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
                printf("  -x <chase_iters>        Set pointer-chase inner iterations (> 0)\n");
                printf("  -l <loads_per_iter>     Set pointer-chase loads per inner iteration (> 0)\n");
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
}

int main(int argc, char *argv[])
{
    
    int BytesPerWord, k;
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
        .chase_iterations       = 5000,
        .chase_loads_per_iter   = 64,
        .walk_file_path         = "array.dat",
        .thread0_pointer_chase  = 0,
        .warmup_iterations      = 4,
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
    long long chase_array_elems     = opts.chase_array_elems;
    ssize_t chase_array_bytes       = 0;
    int chase_iterations            = opts.chase_iterations;
    int chase_loads_per_iter        = opts.chase_loads_per_iter;
    const char *walk_file_path      = opts.walk_file_path;
    int thread0_pointer_chase       = opts.thread0_pointer_chase;
    int warmup_iterations           = opts.warmup_iterations;
    struct pointer_chase_line *chase_array = NULL;
    volatile uint64_t chase_sink = 0;
    uint64_t pointer_chase_total_cycles = 0;
    unsigned long long pointer_chase_total_loads = 0ULL;
    uint64_t pointer_chase_measure_window_cycles = 0;
    uint64_t pointer_chase_iter_window_cycles_total = 0;
    uint64_t pointer_chase_next_offset = 0;
    unsigned long long stream_measured_iterations = 0ULL;
    int observed_thread_count = 1;
    int observed_stream_worker_count = 0;
    volatile int stream_iter_done = 0;
    int stream_iter_workers_remaining = 0;
    uint64_t arch_timer_hz = cycles_per_second();

    if (debug_enabled)
        {
            char dbg_msg[512];
            snprintf(dbg_msg, sizeof(dbg_msg),
                "Command line arguments: rd_percentage=%d, pause=%d, array_size=%lld, iterations=%d, periodic_stats_ticks=%lld, skip_init=%d, m5_enabled=%d, debug_enabled=%d, chase_nodes=%lld, chase_iterations=%d, chase_loads_per_iter=%d, walk_file=%s, thread0_pointer_chase=%d, warmup_iterations=%d",
                rd_percentage, pause, STREAM_ARRAY_SIZE, run_iterations,
                periodic_stats_ticks, cli_skip_init, m5_enabled, debug_enabled,
                chase_array_elems, chase_iterations, chase_loads_per_iter, walk_file_path, thread0_pointer_chase, warmup_iterations);
            debug_log_json(dbg_msg);
        }

    if (thread0_pointer_chase && warmup_iterations >= run_iterations)
    {
        printf("ERROR: with -t 1, warmup iterations (-u=%d) must be smaller than total iterations (-n=%d).\n",
               warmup_iterations, run_iterations);
        printf("Please choose values such that measured iterations are explicit and stable (example: -n 30 -u 10).\n");
        exit(-1);
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
    if (chase_array_elems == 0)
        chase_array_elems = (long long)(POINTER_CHASE_DEFAULT_BYTES / POINTER_CHASE_CACHE_LINE);
    chase_array_bytes = (ssize_t)chase_array_elems * (ssize_t)sizeof(struct pointer_chase_line);
    k = posix_memalign((void **)&chase_array, POINTER_CHASE_CACHE_LINE, (size_t)chase_array_bytes);
    if (k != 0)
    {
        printf("Allocation of pointer-chase array failed, return code is %d\n",k);
        exit(1);
    }
    init_pointer_walk(walk_file_path, chase_array, (uint64_t)chase_array_elems);

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
        printf("Pointer-chase array elements = %lld (cache-line nodes)\n", chase_array_elems);
        printf("Pointer-chase array memory = %.1f MiB (= %.1f GiB).\n",
          ((double)chase_array_bytes) / 1024.0 / 1024.0,
          ((double)chase_array_bytes) / 1024.0 / 1024.0 / 1024.0);
        printf("Pointer-chase config: thread0=%s chase_iters=%d chase_loads_per_iter=%d walk_file='%s'\n",
               thread0_pointer_chase ? "enabled" : "disabled",
               chase_iterations, chase_loads_per_iter, walk_file_path);

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
        int stream_worker_count = 0;
        int stream_worker_idx = -1;
        int iter;
        int effective_warmup_iters = warmup_iterations;
        int measured_iters = run_iterations;
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

        stream_worker_count = thread0_pointer_chase ?
                              (thread_count > 1 ? thread_count - 1 : 0) :
                              thread_count;
        if (thread_id == 0)
        {
            observed_thread_count = thread_count;
            observed_stream_worker_count = stream_worker_count;
        }

        if (stream_worker_count > 0)
        {
            if (!thread0_pointer_chase)
                stream_worker_idx = thread_id;
            else if (thread_id > 0)
                stream_worker_idx = thread_id - 1;
        }

        // Per-thread partitioning of the total_blocks across thread_count
        // OpenMP workers that run STREAM copy, distributing any remainder
        // one block at a time to the lowest-numbered worker indices.
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
        if (debug_enabled && thread_id < 4)
            debug_log_json("Computed thread partition for STREAM kernel");

        if (effective_warmup_iters >= run_iterations)
            effective_warmup_iters = (run_iterations > 0) ? (run_iterations - 1) : 0;
        if (effective_warmup_iters < 0)
            effective_warmup_iters = 0;
        measured_iters = run_iterations - effective_warmup_iters;
        (void)thread_count;

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

        if (thread0_pointer_chase)
        {
            for (iter = 0; iter < effective_warmup_iters; iter++)
            {
                if (thread_id > 0 && local_elements > 0)
                    STREAM_copy_rw(a + local_start, b + local_start, &local_elements, &pause);
#ifdef _OPENMP
                #pragma omp barrier
#endif
            }
            stream_measured_iterations = (unsigned long long)measured_iters;

#ifdef _OPENMP
            #pragma omp barrier
#endif
            if (thread_id == 0)
                pointer_chase_measure_window_cycles = now_cycles();

            for (iter = 0; iter < measured_iters; iter++)
            {
                uint64_t iter_chase_cycles_before = pointer_chase_total_cycles;
                unsigned long long iter_chase_loads_before = pointer_chase_total_loads;
                uint64_t chase_kernel_calls_this_iter = 0;
                uint64_t iter_window_begin_cycles = 0;
                uint64_t iter_window_cycles = 0;
                double chase_duty_pct = 0.0;

                if (thread_id == 0)
                    iter_window_begin_cycles = now_cycles();

#ifdef _OPENMP
                #pragma omp single
#endif
                {
                    stream_iter_workers_remaining = stream_worker_count;
                    stream_iter_done = (stream_worker_count == 0) ? 1 : 0;
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif

                if (thread_id == 0)
                {
                    while (1)
                    {
                        uint64_t chase_begin_cycles = now_cycles();
                        uint64_t kernel_cycles = 0;
                        uint64_t chase_value = pointer_chase_kernel(chase_array,
                                                                    (uint64_t)chase_array_elems,
                                                                    chase_iterations,
                                                                    chase_loads_per_iter,
                                                                    &pointer_chase_next_offset,
                                                                    &kernel_cycles);
                        uint64_t chase_end_cycles = now_cycles();
                        chase_sink ^= chase_value;
                        pointer_chase_total_cycles += kernel_cycles;
                        pointer_chase_total_loads += (unsigned long long)chase_iterations *
                                                     (unsigned long long)chase_loads_per_iter;
                        chase_kernel_calls_this_iter++;
#ifdef _OPENMP
                        #pragma omp flush(stream_iter_done)
#endif
                        if (stream_iter_done)
                            break;
                        (void)chase_begin_cycles;
                        (void)chase_end_cycles;
                    }
                }
                else if (local_elements > 0)
                {
                    if (debug_enabled && iter == 0 && thread_id < 2)
                        debug_log_json("Thread entering synchronized STREAM/pointer-chase measurement loop");
                    STREAM_copy_rw(a + local_start, b + local_start, &local_elements, &pause);
#ifdef _OPENMP
                    if (stream_worker_count > 0)
                    {
                        int remaining_after = 0;
                        #pragma omp atomic capture
                        remaining_after = --stream_iter_workers_remaining;
                        if (remaining_after == 0)
                        {
                            stream_iter_done = 1;
                            #pragma omp flush(stream_iter_done)
                        }
                    }
#endif
                }
#ifdef _OPENMP
                #pragma omp barrier
#endif
                if (thread_id == 0)
                {
                    uint64_t iter_cycles =
                        pointer_chase_total_cycles - iter_chase_cycles_before;
                    iter_window_cycles = now_cycles() - iter_window_begin_cycles;
                    pointer_chase_iter_window_cycles_total += iter_window_cycles;
                    if (iter_window_cycles > 0ULL)
                        chase_duty_pct = ((double)iter_cycles * 100.0) / (double)iter_window_cycles;
                    if (iter < 2)
                    {
                        char dbg_data[256];
                        snprintf(dbg_data, sizeof(dbg_data),
                                 "{\"iter\":%d,\"iter_chase_kernel_calls\":%llu,\"iter_loads\":%llu,"
                                 "\"iter_chase_duty_pct\":%.4f}",
                                 iter,
                                 (unsigned long long)chase_kernel_calls_this_iter,
                                 (unsigned long long)(pointer_chase_total_loads - iter_chase_loads_before),
                                 chase_duty_pct);
                        // #region agent log H9 continuous-chase verification
                        debug_emit_stdout("post-fix", "H9_continuous_chase_per_iter",
                                          "stream_omp.c:measurement_iter",
                                          "Continuous chase activity while STREAM iteration executes", dbg_data);
                        // #endregion
                    }
                }
            }

            if (thread_id == 0)
                pointer_chase_measure_window_cycles = now_cycles() - pointer_chase_measure_window_cycles;
        }
        else
        {
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
        }

#ifdef _OPENMP
        #pragma omp barrier
        if (debug_enabled && thread_id < 4)
            debug_log_json("Thread reached final ROI barrier");
        #pragma omp master
#endif
        {
            if (thread0_pointer_chase)
            {
                char ptr_dbg_msg[256];
                double latency_sim_ns = 0.0;
                double latency_cycles = 0.0;
                double measured_bw_mb_s = 0.0;
                if (pointer_chase_total_loads > 0ULL && pointer_chase_total_cycles > 0ULL)
                    latency_cycles = (double)pointer_chase_total_cycles /
                                     (double)pointer_chase_total_loads;
                if (latency_cycles > 0.0 && arch_timer_hz > 0ULL)
                    latency_sim_ns = latency_cycles * (1.0e9 / (double)arch_timer_hz);
                if (stream_worker_count > 0 &&
                    pointer_chase_measure_window_cycles > 0ULL &&
                    arch_timer_hz > 0ULL &&
                    stream_measured_iterations > 0ULL)
                {
                    double measured_bytes = (double)stream_measured_iterations *
                                            (double)array_elements *
                                            (double)sizeof(STREAM_TYPE) *
                                            2.0;
                    measured_bw_mb_s = measured_bytes /
                                       ((double)pointer_chase_measure_window_cycles / (double)arch_timer_hz) /
                                       1.0e6;
                }
                {
                    double overall_chase_duty_pct = 0.0;
                    double avg_chase_kernels_per_iter = 0.0;
                    double chase_loads_per_sec = 0.0;
                    double pointer_chase_bw_mb_s = 0.0;
                    double combined_bw_mb_s = 0.0;
                    if (pointer_chase_iter_window_cycles_total > 0ULL)
                        overall_chase_duty_pct =
                            ((double)pointer_chase_total_cycles * 100.0) /
                            (double)pointer_chase_iter_window_cycles_total;
                    if (stream_measured_iterations > 0ULL &&
                        chase_iterations > 0 && chase_loads_per_iter > 0)
                        avg_chase_kernels_per_iter =
                            ((double)pointer_chase_total_loads /
                             ((double)chase_iterations * (double)chase_loads_per_iter)) /
                            (double)stream_measured_iterations;
                    if (pointer_chase_measure_window_cycles > 0ULL && arch_timer_hz > 0ULL)
                    {
                        chase_loads_per_sec =
                            (double)pointer_chase_total_loads /
                            ((double)pointer_chase_measure_window_cycles / (double)arch_timer_hz);
                        pointer_chase_bw_mb_s =
                            ((double)pointer_chase_total_loads * 8.0) /
                            ((double)pointer_chase_measure_window_cycles / (double)arch_timer_hz) /
                            1.0e6;
                    }
                    combined_bw_mb_s = measured_bw_mb_s + pointer_chase_bw_mb_s;
                    // #region agent log H5 aggregate duty-cycle evidence
                    {
                        char duty_data[640];
                        snprintf(duty_data, sizeof(duty_data),
                                 "{\"overall_chase_duty_pct\":%.4f,\"iter_window_cycles_total\":%llu,"
                                 "\"total_chase_cycles\":%llu,\"measured_iters\":%llu,"
                                 "\"avg_chase_kernels_per_iter\":%.6f,"
                                 "\"thread_count\":%d,\"stream_worker_count\":%d,"
                                 "\"chase_loads_per_sec\":%.6f,"
                                 "\"pointer_chase_bw_MB_s\":%.6f,\"combined_bw_MB_s\":%.6f}",
                                 overall_chase_duty_pct,
                                 (unsigned long long)pointer_chase_iter_window_cycles_total,
                                 (unsigned long long)pointer_chase_total_cycles,
                                 stream_measured_iterations,
                                 avg_chase_kernels_per_iter,
                                 observed_thread_count,
                                 observed_stream_worker_count,
                                 chase_loads_per_sec,
                                 pointer_chase_bw_mb_s,
                                 combined_bw_mb_s);
                        debug_emit_stdout("post-fix", "H5_aggregate_chase_duty",
                                          "stream_omp.c:final_summary",
                                          "Aggregate chase duty cycle across measured window", duty_data);
                    }
                    // #endregion
                }
                snprintf(ptr_dbg_msg, sizeof(ptr_dbg_msg),
                         "Pointer-chase: sink=%llu total_cycles=%llu total_loads=%llu avg_latency_cycles=%.6f avg_latency_ns=%.6f measured_bw_MB_s=%.6f warmup_iters=%d measured_iters=%llu timer_hz=%llu",
                         (unsigned long long)chase_sink,
                         (unsigned long long)pointer_chase_total_cycles,
                         pointer_chase_total_loads,
                         latency_cycles,
                         latency_sim_ns,
                         measured_bw_mb_s,
                         effective_warmup_iters,
                         stream_measured_iterations,
                         (unsigned long long)arch_timer_hz);
                debug_log_json(ptr_dbg_msg);
                {
                    char dbg_data[384];
                    snprintf(dbg_data, sizeof(dbg_data),
                             "{\"total_cycles\":%llu,\"total_loads\":%llu,\"avg_latency_ns\":%.6f,"
                             "\"avg_latency_cycles\":%.6f,\"measured_bw_MB_s\":%.6f,"
                             "\"measured_iters\":%llu,\"timer_hz\":%llu}",
                             (unsigned long long)pointer_chase_total_cycles,
                             pointer_chase_total_loads,
                             latency_sim_ns,
                             latency_cycles,
                             measured_bw_mb_s,
                             stream_measured_iterations,
                             (unsigned long long)arch_timer_hz);
                    // #region agent log H3 summary correlation evidence
                    debug_emit_stdout("post-fix", "H3_summary_bw_latency_correlation",
                                      "stream_omp.c:final_summary",
                                      "Final measured BW and latency summary", dbg_data);
                    // #endregion
                }
                if (stream_worker_count > 0)
                    printf("%.6f %.6f\n", measured_bw_mb_s, latency_sim_ns);
            }
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
    free(chase_array);
    return(0);
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/time.h>
#include <omp.h>
#include "aux.h"

#define HLINE "-------------------------------------------------------------\n"

static long long debug_now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((long long)tv.tv_sec * 1000LL) + ((long long)tv.tv_usec / 1000LL);
}

void debug_log_json(const char *message)
{
    fprintf(stdout, "{\"message\":\"%s\",\"timestamp\":%lld}\n", message, debug_now_ms());
    fflush(stdout);
}

uint64_t now_cycles(void)
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

uint64_t cycles_per_second(void)
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

void m5_dump_reset_stats(uint64_t delay, uint64_t period)
{
#if defined(__aarch64__)
    register uint64_t x0 __asm__("x0") = delay;
    register uint64_t x1 __asm__("x1") = period;
    __asm__ __volatile__(
        ".inst 0xFF420110\n\t"
        : "+r"(x0), "+r"(x1));
#endif
}

void m5_exit(uint64_t delay)
{
#if defined(__aarch64__)
    register uint64_t x0 __asm__("x0") = delay;
    __asm__ __volatile__(
        ".inst 0xFF210110\n\t"
        : "+r"(x0));
#endif
}

void m5_dump_stats(uint64_t delay, uint64_t period)
{
#if defined(__aarch64__)
    register uint64_t x0 __asm__("x0") = delay;
    register uint64_t x1 __asm__("x1") = period;
    __asm__ __volatile__(
        ".inst 0xFF410110\n\t"
        : "+r"(x0), "+r"(x1));
#endif
}

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
        // READ THE OFFSET FROM THE FILE
        if (fscanf(input_file, "%llu", &tmp) != 1)
        {
            fclose(input_file);
            return -1;
        }
        // CHECK IF THE OFFSET IS VALID
        // - It must be a multiple of the cache line size
        // - It must be less than the maximum offset
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

void init_pointer_walk(const char *walk_file_path,
                       struct pointer_chase_line *walk_array,
                       uint64_t elems)
{   
    // LOAD THE POINTER WALK FROM THE FILE
    if (load_pointer_walk_file(walk_file_path, walk_array, elems) == 0)
    {
        printf("Pointer walk loaded from '%s'.\n", walk_file_path);
        return;
    }

    // IF THE FILE IS NOT AVAILABLE OR INVALID, GENERATE A DETERMINISTIC WALK IN-MEMORY
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

uint64_t pointer_chase_kernel(struct pointer_chase_line *walk_array,
                              uint64_t elems,
                              uint64_t total_loads,
                              uint64_t *next_offset_state,
                              uint64_t *kernel_cycles_out)
{
    uint64_t next_offset = 0;
    uint64_t base_addr_u64;
    uint64_t max_offset;

    if (walk_array == NULL || elems == 0 || total_loads == 0 ||
        next_offset_state == NULL)
        return 0;

    max_offset = elems * POINTER_CHASE_CACHE_LINE;
    if (*next_offset_state < max_offset && ((*next_offset_state % POINTER_CHASE_CACHE_LINE) == 0))
        next_offset = *next_offset_state;

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
    #error "pointer_chase_kernel requires AArch64 inline assembly."
#endif

    *next_offset_state = next_offset;
    return next_offset;
}

void print_initial_kernel_info(int bytes_per_word,
                               long long stream_array_size,
                               long long chase_array_elems,
                               ssize_t chase_array_bytes,
                               int thread0_pointer_chase,
                               uint64_t chase_total_loads,
                               const char *walk_file_path,
                               int run_iterations)
{
    printf(HLINE);
    printf("$ Memory bandwidth load kernel $\n");
    printf(HLINE);
    printf("This system uses %d bytes per array element.\n", bytes_per_word);

    printf("Total Aggregate Array size = %llu (elements)\n", (unsigned long long)stream_array_size);
    printf("Total Aggregate Memory per array = %.1f MiB (= %.1f GiB).\n",
           bytes_per_word * ((double)stream_array_size / 1024.0 / 1024.0),
           bytes_per_word * ((double)stream_array_size / 1024.0 / 1024.0 / 1024.0));
    printf("Total Aggregate memory required = %.1f MiB (= %.1f GiB).\n",
           (2.0 * bytes_per_word) * ((double)stream_array_size / 1024.0 / 1024.0),
           (2.0 * bytes_per_word) * ((double)stream_array_size / 1024.0 / 1024.0 / 1024.0));
    printf("Pointer-chase array elements = %lld (cache-line nodes)\n", chase_array_elems);
    printf("Pointer-chase array memory = %.1f MiB (= %.1f GiB).\n",
           ((double)chase_array_bytes) / 1024.0 / 1024.0,
           ((double)chase_array_bytes) / 1024.0 / 1024.0 / 1024.0);
    printf("Pointer-chase config: thread0=%s chase_total_loads=%llu walk_file='%s'\n",
           thread0_pointer_chase ? "enabled" : "disabled",
           (unsigned long long)chase_total_loads, walk_file_path);

    printf(HLINE);
    printf("The kernel will be executed %d times.\n", run_iterations);

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel
    {
#pragma omp master
        {
            int thread_count = omp_get_num_threads();
            printf("Number of Threads requested = %i\n", thread_count);
        }
    }
#endif

#ifdef _OPENMP
    {
        int thread_count = 0;
#pragma omp parallel
        {
#pragma omp atomic
            thread_count++;
            printf("Number of Threads counted = %i\n", thread_count);
        }
    }
#endif
}

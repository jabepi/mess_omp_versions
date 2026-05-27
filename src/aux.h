#ifndef AUX_H
#define AUX_H

#include <stdint.h>
#include <sys/types.h>

#define POINTER_CHASE_CACHE_LINE 128
#define POINTER_CHASE_DEFAULT_BYTES (100ULL * 1024ULL * 1024ULL)

struct pointer_chase_line
{
    uint64_t next_offset;
    uint8_t pad[POINTER_CHASE_CACHE_LINE - sizeof(uint64_t)];
};

void debug_log_json(const char *message);
uint64_t now_cycles(void);
uint64_t cycles_per_second(void);

void m5_dump_reset_stats(uint64_t delay, uint64_t period);
void m5_exit(uint64_t delay);
void m5_dump_stats(uint64_t delay, uint64_t period);

void init_pointer_walk(const char *walk_file_path,
                       struct pointer_chase_line *walk_array,
                       uint64_t elems);

uint64_t pointer_chase_kernel(struct pointer_chase_line *walk_array,
                              uint64_t elems,
                              uint64_t total_loads,
                              uint64_t *next_offset_state,
                              uint64_t *kernel_cycles_out);

void print_initial_kernel_info(int bytes_per_word,
                               long long stream_array_size,
                               long long chase_array_elems,
                               ssize_t chase_array_bytes,
                               int thread0_pointer_chase,
                               uint64_t chase_total_loads,
                               const char *walk_file_path,
                               int run_iterations);

#endif

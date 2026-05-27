# Parallel ROI Block (stream_omp.c)

This note explains the main OpenMP region in `stream_omp.c` (the block currently around line 483 to the end of `main`), where STREAM traffic and pointer-chase latency measurement are coordinated.

## What this block does

Inside one OpenMP parallel region, the code:

1. Splits STREAM work across threads.
2. Optionally reserves thread 0 for pointer-chase (`-t 1` mode).
3. Runs warmup iterations.
4. Starts a synchronized measurement window.
5. Runs pointer-chase and STREAM concurrently.
6. Stops STREAM workers when pointer-chase is done.
7. Aggregates and prints final pointer-chase latency output.

## Per-thread setup

At region entry, each thread computes:

- `thread_id`, `thread_count`
- `stream_worker_count`: either all threads, or all except thread 0 in pointer-chase mode
- `stream_worker_idx`: compact worker index used for partitioning STREAM slices

Then the code partitions the stream arrays in units of `STREAM_KERNEL_GRAIN_ELEMS` blocks:

- `total_blocks`: total number of fixed-size blocks
- `chunk`: base blocks per STREAM worker
- `remainder`: extra blocks distributed one-by-one to low worker indices
- `local_start`, `local_elements`: per-thread slice in `a` and `b`

This guarantees deterministic contiguous slices per worker and no overlap.

## Warmup and measured iterations

- `warmup_iters = opts.warmup_iterations`
- `measured_iters = opts.run_iterations`

Warmup and measured iterations are independent.

## Gem5 stats control point

After an initial barrier, the OpenMP master optionally:

- resets stats: `m5_dump_reset_stats`
- enables periodic dumps: `m5_dump_stats(..., opts.periodic_stats_ticks)`

Then a second barrier ensures all threads begin the measurement phase consistently.

## Two execution modes

## 1) Pointer-chase mode (`opts.thread0_pointer_chase == 1`)

### Phase 1: STREAM warmup

- Non-zero threads (STREAM workers) run `STREAM_copy_rw` on their local slices.
- Thread 0 does not run STREAM in this mode.
- Barrier each iteration keeps warmup synchronized.

### Phase 2: Handshake + overlap run

Then:

- **Thread 0** runs fixed pointer-chase work for `measured_iters`:
  - calls `pointer_chase_kernel(...)`
  - accumulates `pointer_chase_total_cycles`
  - accumulates `pointer_chase_total_loads`
  - updates `chase_sink` to preserve dependency/side effects

- **STREAM worker threads**:
  - run warmup first
  - set `start_pointer_chase = 1` after warmup (worker 0 in the pool)
  - enter continuous full-pass STREAM loop
  - poll `stream_workers_stop` (with `omp flush`) and exit when set

Thread 0 waits for `start_pointer_chase` before beginning pointer-chase.
When thread 0 finishes pointer-chase loops, it sets `stream_workers_stop = 1`.

## 2) STREAM-only mode (`opts.thread0_pointer_chase == 0`)

All threads run normal STREAM iterations (`opts.run_iterations`) over their local slices.
If debug is enabled, thread 0 and some workers emit progress logs.

## Final reduction/output

After the final barrier, OpenMP master computes and reports:

- `latency_cycles = pointer_chase_total_cycles / pointer_chase_total_loads`
- `latency_sim_ns = latency_cycles * (1e9 / arch_timer_hz)`
- `total_latency_ns = pointer_chase_total_cycles * (1e9 / arch_timer_hz)`

If pointer-chase mode was active, it prints:

- `"Pointer-chase latency: total_ns=<...> avg_ns_per_access=<...>"`

This is the machine-readable result consumed by your scripts.

If gem5 mode is enabled, it also performs a final `m5_dump_stats`, and after the parallel region ends it calls `m5_exit(0)`.

## Why this structure matters

- Barriers align the start/end of measurement.
- Thread 0 specialization avoids mixing STREAM work with pointer-chase in the same thread.
- A start flag (`start_pointer_chase`) ensures pointer-chase starts only after STREAM warmup completes.


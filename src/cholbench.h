#ifndef _CHOLBENCH_
#define _CHOLBENCH_

#ifdef _cplusplus
extern "C" {
#endif

typedef enum {
  CHOLBENCH_SOLVER_CUSOLVER = 0,
  CHOLBENCH_SOLVER_HYPRE = 1
} cholbench_solver_t;

typedef enum {
  CHOLBENCH_PRECISION_FP64 = 0,
  CHOLBENCH_PRECISION_FP32 = 1,
  CHOLBENCH_PRECISION_FP16 = 2
} cholbench_precision_t;

typedef enum {
  CHOLBENCH_ORDERING_NONE = -1,
  CHOLBENCH_ORDERING_RCM = 0,
  CHOLBENCH_ORDERING_AMD = 1,
  CHOLBENCH_ORDERING_METIS = 2
} cholbench_ordering_t;

struct cholbench;
struct cholbench *cholbench_init(int argc, char *argv[]);

struct csr;
struct csr *cholbench_matrix_read(const struct cholbench *cb);
void cholbench_matrix_print(const struct csr *A);
void cholbench_matrix_free(struct csr *A);

void cholbench_bench(struct csr *A, const struct cholbench *cb);

void cholbench_finalize(struct cholbench *cb);

#ifdef _cplusplus
}
#endif

#endif

#ifndef _LSBENCH_
#define _LSBENCH_

#ifdef _cplusplus
extern "C" {
#endif

typedef enum {
  LSBENCH_SOLVER_NONE = -1,
  LSBENCH_SOLVER_CUSOLVER = 0,
  LSBENCH_SOLVER_HYPRE = 1,
  LSBENCH_SOLVER_AMGX = 2
} lsbench_solver_t;

typedef enum {
  LSBENCH_PRECISION_FP64 = 0,
  LSBENCH_PRECISION_FP32 = 1,
  LSBENCH_PRECISION_FP16 = 2
} lsbench_precision_t;

typedef enum {
  LSBENCH_ORDERING_RCM = 0,
  LSBENCH_ORDERING_AMD = 1,
  LSBENCH_ORDERING_METIS = 2
} lsbench_ordering_t;

struct lsbench;
struct lsbench *lsbench_init(int argc, char *argv[]);

struct csr;
struct csr *lsbench_matrix_read(const struct lsbench *cb);
void lsbench_matrix_print(const struct csr *A);
void lsbench_matrix_free(struct csr *A);

void lsbench_bench(struct csr *A, const struct lsbench *cb);

void lsbench_finalize(struct lsbench *cb);

#ifdef _cplusplus
}
#endif

#endif

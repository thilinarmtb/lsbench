#ifndef _LSBENCH_
#define _LSBENCH_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  LSBENCH_SOLVER_NONE = -1,
  LSBENCH_SOLVER_CUSOLVER = 0,
  LSBENCH_SOLVER_HYPRE = 1,
  LSBENCH_SOLVER_AMGX = 2,
  LSBENCH_SOLVER_CHOLMOD = 3,
  LSBENCH_SOLVER_PARALMOND = 4
} lsbench_solver_t;

typedef enum {
  LSBENCH_PRECISION_FP64 = 0,
  LSBENCH_PRECISION_FP32 = 1,
  LSBENCH_PRECISION_FP16 = 2
} lsbench_precision_t;

typedef enum {
  LSBENCH_ORDERING_NONE = -1,
  LSBENCH_ORDERING_RCM = 0,
  LSBENCH_ORDERING_AMD = 1,
  LSBENCH_ORDERING_METIS = 2
} lsbench_ordering_t;

struct csr;
struct csr *lsbench_matrix_read(const char *fname);
void csr_symA(struct csr *A);
void csr_spmv(const double a, const struct csr *A, const double *x,
              const double b, double *y);
double l2norm(const double *x, const int n);
double glmax(const double *x, const int n);
double glmin(const double *x, const int n);
double glamax(const double *x, const int n);

void lsbench_matrix_print(const struct csr *A);
void lsbench_matrix_free(struct csr *A);

struct lsbench;
struct lsbench *lsbench_init(int argc, char *argv[]);
const char *lsbench_get_matrix_name(struct lsbench *cb);
void lsbench_bench(struct csr *A, const struct lsbench *cb);
void lsbench_finalize(struct lsbench *cb);

#ifdef __cplusplus
}
#endif

#endif

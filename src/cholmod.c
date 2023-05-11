#include "lsbench-impl.h"

#if defined(LSBENCH_CHOLMOD)
#include <cholmod.h>

static int initialized = 0;
static cholmod_common cm;

// Callback function to call if an error occurs.
static void err_handler(int status, const char *file, int line,
                        const char *message) {
  printf("cholmod error: file: %s line: %d status: %d: %s\n", file, line,
         status, message);
}

struct cholmod_csr {
  unsigned nr;
  cholmod_sparse *A;
  cholmod_factor *L;
  cholmod_dense *r;
};

#define idx_t int64_t

#define allocate_triplet cholmod_l_allocate_triplet
#define triplet_to_sparse cholmod_l_triplet_to_sparse
#define free_triplet cholmod_l_free_triplet
#define analyze cholmod_l_analyze
#define factorize cholmod_l_factorize
#define zeros cholmod_l_zeros
#define solve cholmod_l_solve
#define free_dense cholmod_l_free_dense
#define finish cholmod_l_finish
#define gpu_stats cholmod_l_gpu_stats
#define change_factor cholmod_l_change_factor

#include "cholmod-impl.h"

#undef allocate_triplet
#undef triplet_to_sparse
#undef free_triplet
#undef analyze
#undef factorize
#undef zeros
#undef solve
#undef free_dense
#undef finish
#undef gpu_stats
#undef change_factor

static void csr_finalize(struct cholmod_csr *A) {
  if (A) {
    cholmod_free_sparse(&A->A, &cm);
    cholmod_free_factor(&A->L, &cm);
    cholmod_free_dense(&A->r, &cm);
  }
  tfree(A);
}

int cholmod_init() {
  if (initialized)
    return 1;

  cholmod_l_start(&cm);
  cm.dtype = CHOLMOD_DOUBLE;
  if (1) {
    cm.itype = CHOLMOD_LONG;
    cm.useGPU = 0;
  }
  cm.error_handler = err_handler;
  initialized = 1;
  return 0;
}
#else
int cholmod_init() { return 1; }
int cholmod_finalize() { return 1; }
int cholmod_bench(double *x, struct csr *A, const double *r,
                  const struct lsbench *cb) {
  return 1;
}
#endif

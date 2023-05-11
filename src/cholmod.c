#include "lsbench-impl.h"
#include <err.h>

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

#define TOKEN_PASTE_(a, b) a##b
#define TOKEN_PASTE(a, b) TOKEN_PASTE_(a, b)
#define SUFFIXED_NAME(name) TOKEN_PASTE(name, SUFFIX)

#define idx_t int64_t
#define SUFFIX _int64
#define allocate_triplet cholmod_l_allocate_triplet
#define triplet_to_sparse cholmod_l_triplet_to_sparse
#define free_triplet cholmod_l_free_triplet
#define analyze cholmod_l_analyze
#define factorize cholmod_l_factorize
#define zeros cholmod_l_zeros
#define solve cholmod_l_solve
#define free_dense cholmod_l_free_dense
#define copy_dense cholmod_l_copy_dense
#define norm_dense cholmod_l_norm_dense
#define free_sparse cholmod_l_free_sparse
#define free_factor cholmod_l_free_factor
#define gpu_stats cholmod_l_gpu_stats
#define change_factor cholmod_l_change_factor
#define sdmult cholmod_l_sdmult

#include "cholmod-impl.h"

#undef idx_t
#undef SUFFIX
#undef allocate_triplet
#undef triplet_to_sparse
#undef free_triplet
#undef analyze
#undef factorize
#undef zeros
#undef solve
#undef free_dense
#undef copy_dense
#undef norm_dense
#undef free_sparse
#undef free_factor
#undef gpu_stats
#undef change_factor
#undef sdmult

#define idx_t int32_t
#define SUFFIX _int32
#define allocate_triplet cholmod_allocate_triplet
#define triplet_to_sparse cholmod_triplet_to_sparse
#define free_triplet cholmod_free_triplet
#define analyze cholmod_analyze
#define factorize cholmod_factorize
#define zeros cholmod_zeros
#define solve cholmod_solve
#define free_dense cholmod_free_dense
#define copy_dense cholmod_copy_dense
#define norm_dense cholmod_norm_dense
#define free_sparse cholmod_free_sparse
#define free_factor cholmod_free_factor
#define gpu_stats cholmod_gpu_stats
#define change_factor cholmod_change_factor
#define sdmult cholmod_sdmult

#include "cholmod-impl.h"

#undef idx_t
#undef SUFFIX
#undef allocate_triplet
#undef triplet_to_sparse
#undef free_triplet
#undef analyze
#undef factorize
#undef zeros
#undef solve
#undef free_dense
#undef copy_dense
#undef norm_dense
#undef free_sparse
#undef free_factor
#undef gpu_stats
#undef change_factor
#undef sdmult

#undef SUFFIXED_NAME
#undef TOKEN_PASTE
#undef TOKEN_PASTE_

int cholmod_init() {
  if (initialized)
    return 1;

  int itype = CHOLMOD_INT;
  switch (itype) {
  case CHOLMOD_INT:
    cholmod_start(&cm);
    break;
  case CHOLMOD_LONG:
    cholmod_l_start(&cm);
    break;
  default:
    errx(EXIT_FAILURE, "Invalid precision for index type !");
    break;
  }

  cm.dtype = CHOLMOD_DOUBLE;
  cm.itype = itype;
  cm.useGPU = (cm.itype == CHOLMOD_LONG ? 1 : 0);
  cm.error_handler = err_handler;

  initialized = 1;

  return 0;
}

int cholmod_finalize() {
  if (!initialized)
    return 1;

  switch (cm.itype) {
  case CHOLMOD_INT:
    cholmod_finish(&cm);
    break;
  case CHOLMOD_LONG:
    cholmod_l_finish(&cm);
    break;
  default:
    errx(EXIT_FAILURE, "Invalid precision for index type !");
    break;
  }

  initialized = 0;

  return 0;
}

int cholmod_bench(double *x, struct csr *A, const double *r,
                  const struct lsbench *cb) {
  if (!initialized)
    return 1;

  switch (cm.itype) {
  case CHOLMOD_INT:
    bench_int32(x, A, r, cb);
    break;
  case CHOLMOD_LONG:
    bench_int64(x, A, r, cb);
    break;
  default:
    errx(EXIT_FAILURE, "Invalid precision for index type !");
    break;
  }
}
#else
int cholmod_init() { return 1; }
int cholmod_finalize() { return 1; }
int cholmod_bench(double *x, struct csr *A, const double *r,
                  const struct lsbench *cb) {
  return 1;
}
#endif

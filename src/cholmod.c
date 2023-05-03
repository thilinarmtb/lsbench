#include "lsbench-impl.h"

#if defined(ENABLE_CHOLMOD)
#include <cholmod.h>

static int initialized = 0;
struct cholmod_csr {
  unsigned nr;
  cholmod_common cm;
  cholmod_sparse *A;
  cholmod_factor *L;
  cholmod_dense *r;
};

int cholmod_init() {}
void cholmod_bench(double *x, struct csr *A, const double *r,
                   const struct lsbench *cb) {}
int cholmod_finalize() {}
#else
int cholmod_init() {}
void cholmod_bench(double *x, struct csr *A, const double *r,
                   const struct lsbench *cb) {}
int cholmod_finalize() {}
#endif

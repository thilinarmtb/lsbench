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
int cholmod_finalize() {}
void cholmod_bench(double *x, struct csr *A, const double *r,
                   const struct lsbench *cb) {}
#else
int cholmod_init() { return 1; }
int cholmod_finalize() { return 1; }
void cholmod_bench(double *x, struct csr *A, const double *r,
                   const struct lsbench *cb) {}
#endif

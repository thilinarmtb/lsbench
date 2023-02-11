#ifndef _CHOLBENCH_IMPL_
#define _CHOLBENCH_IMPL_

#include "cholbench.h"
#include <stdlib.h>

#ifdef _cplusplus
extern "C" {
#endif

struct csr {
  unsigned nrows, base;
  unsigned *offs, *cols;
  double *vals;
  void *ptr;
};

#define tcalloc(T, n) (T *)calloc(n, sizeof(T))

static inline void sfree(void *p, const char *file, unsigned line) {
  if (p)
    free(p);
}
#define tfree(p) sfree((void *)p, __FILE__, __LINE__)

int cusparse_init();
int cusparse_finalize();
void cusparse_bench(double *x, struct csr *A, const double *r,
                    unsigned ntrials);

#ifdef _cplusplus
}
#endif

#endif

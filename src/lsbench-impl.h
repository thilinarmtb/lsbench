#ifndef _LSBENCH_IMPL_
#define _LSBENCH_IMPL_

#include "lsbench.h"
#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

struct lsbench {
  char *matrix;
  lsbench_solver_t solver;
  lsbench_ordering_t ordering;
  lsbench_precision_t precision;
  unsigned verbose, trials;
};

struct csr {
  unsigned nrows, base;
  unsigned *offs, *cols;
  double *vals;
};

#define tcalloc(T, n) (T *)calloc(n, sizeof(T))

static inline void sfree(void *p, const char *file, unsigned line) {
  if (p)
    free(p);
}
#define tfree(p) sfree((void *)p, __FILE__, __LINE__)

struct backend {
  int (*init)();
  int (*finalize)();
  void (*bench)(double *, struct csr *, const double *, const struct lsbench *);
};

int cusparse_init();
int cusparse_finalize();
int cusparse_bench(double *x, struct csr *A, const double *r,
                   const struct lsbench *cb);

int hypre_init();
int hypre_finalize();
int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb);

int amgx_init();
int amgx_finalize();
int amgx_bench(double *x, struct csr *A, const double *r,
               const struct lsbench *cb);

int cholmod_init();
int cholmod_finalize();
int cholmod_bench(double *x, struct csr *A, const double *r,
                  const struct lsbench *cb);

int paralmond_init();
int paralmond_finalize();
int paralmond_bench(double *x, struct csr *A, const double *r,
                    const struct lsbench *cb);

#ifdef __cplusplus
}
#endif
#endif

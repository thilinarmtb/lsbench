#ifndef _CHOLBENCH_IMPL_
#define _CHOLBENCH_IMPL_

#include "cholbench.h"
#include <stdlib.h>
#if defined(CHOLBENCH_MPI)
#include <mpi.h>
#endif

#ifdef _cplusplus
extern "C" {
#endif

struct cholbench {
  char *matrix;
  cholbench_solver_t solver;
  cholbench_ordering_t ordering;
  cholbench_precision_t precision;
  unsigned verbose, trials;
#if defined(CHOLBENCH_MPI)
  MPI_Comm comm;
#else
  int comm;
#endif
};

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
                    const struct cholbench *cb);

int hypre_init();
int hypre_finalize();
void hypre_bench(double *x, struct csr *A, const double *r,
                 const struct cholbench *cb);

#ifdef _cplusplus
}
#endif

#endif

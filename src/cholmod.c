#include "lsbench-impl.h"

#if defined(LSBENCH_CHOLMOD)
#include <cholmod.h>

static int initialized = 0;
static cholmod_common cm;

struct cholmod_csr {
  unsigned nr;
  cholmod_sparse *A;
  cholmod_factor *L;
  cholmod_dense *r;
};

// Callback function to call if an error occurs.
static void err_handler(int status, const char *file, int line,
                        const char *message) {
  printf("cholmod error: file: %s line: %d status: %d: %s\n", file, line,
         status, message);
}

static struct cholmod_csr *csr_init(const struct csr *A,
                                    const struct lsbench *cb) {
  struct cholmod_csr *B = tcalloc(struct cholmod_csr, 1);

  uint nnz = A->offs[A->nrows];
  cholmod_triplet *T =
      cholmod_allocate_triplet(A->nrows, A->nrows, nnz, -1, CHOLMOD_REAL, &cm);
  int32_t *Ti = (int32_t *)T->i, *Tj = (int32_t *)T->j;

  uint z = 0;
  double *Tx = (double *)T->x;
  for (uint i = 0; i < A->nrows; i++) {
    uint j;
    for (j = A->offs[i]; A->cols[j] - A->base < i; j++)
      ;
    for (uint je = A->offs[i + 1]; j < je; j++)
      Ti[z] = i, Tj[z] = A->cols[j] - A->base, Tx[z] = A->vals[j], z++;
  }
  T->nnz = z;

  // Convert triplet to CSC matrix.
  B->A = cholmod_triplet_to_sparse(T, T->nnz, &cm);
  cholmod_free_triplet(&T, &cm);

  // Factorize.
  B->L = cholmod_analyze(B->A, &cm);
  cholmod_factorize(B->A, B->L, &cm);

  B->r = cholmod_zeros(A->nrows, 1, CHOLMOD_REAL, &cm);
  B->nr = A->nrows;

  return B;
}

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

  cholmod_start(&cm);
  cm.itype = CHOLMOD_INT;
  cm.dtype = CHOLMOD_DOUBLE;
  cm.error_handler = err_handler;
  initialized = 1;
  return 0;
}

int cholmod_finalize() {
  if (!initialized)
    return 1;
  cholmod_finish(&cm);
  initialized = 0;
  return 0;
}

int cholmod_bench(double *x, struct csr *A, const double *r,
                  const struct lsbench *cb) {
  if (!initialized)
    return 1;

  struct cholmod_csr *B = csr_init(A, cb);
  int nr = A->nrows;

  cholmod_dense *cr0 = cholmod_zeros(nr, 1, CHOLMOD_REAL, &cm);
  double *pr = (double *)B->r->x;
  double *pr0 = (double *)cr0->x;
  for (uint i = 0; i < B->nr; i++) {
    pr[i] = r[i];
    pr0[i] = r[i];
  }

  cholmod_dense *xd = cholmod_solve(CHOLMOD_A, B->L, B->r, &cm);

  if (cb->verbose > 0) {
    double one[2] = {1.0, 0}, m1[2] = {-1.0, 0};
    cholmod_dense *rd = cholmod_copy_dense(B->r, &cm);
    cholmod_sdmult(B->A, 0, m1, one, xd, rd, &cm);
    printf("norm(b-Ax) = %e\n", cholmod_norm_dense(rd, 2, &cm));
    cholmod_free_dense(&rd, &cm);
  }

  double *xx = (double *)xd->x;
  for (unsigned i = 0; i < B->nr; i++)
    x[i] = xx[i];

  cholmod_free_dense(&xd, &cm), cholmod_finalize(B);
}
#else
int cholmod_init() { return 1; }
int cholmod_finalize() { return 1; }
int cholmod_bench(double *x, struct csr *A, const double *r,
                  const struct lsbench *cb) {
  return 1;
}
#endif

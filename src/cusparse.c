#include "cholbench-impl.h"
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <err.h>
#include <stdio.h>
#include <time.h>

#define chk_solver(err)                                                        \
  {                                                                            \
    cusolverStatus_t err_ = (err);                                             \
    if (err_ != CUSOLVER_STATUS_SUCCESS)                                       \
      errx(EXIT_FAILURE, "%s:%d cusolverSp error: %s", __FILE__, __LINE__);    \
  }

#define chk_sparse(err)                                                        \
  {                                                                            \
    cusparseStatus_t err_ = (err);                                             \
    if (err_ != CUSPARSE_STATUS_SUCCESS) {                                     \
      errx(EXIT_FAILURE, "%s:%d cusparse error: %s", __FILE__, __LINE__,       \
           cusparseGetErrorString(err_));                                      \
    }                                                                          \
  }

#define chk_rt(err)                                                            \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      errx(EXIT_FAILURE, "%s:%d cuda error: %s", __FILE__, __LINE__,           \
           cudaGetErrorString(err_));                                          \
    }                                                                          \
  }

static int initialized = 0;
static cusolverSpHandle_t solver = NULL;
static cusparseHandle_t sparse = NULL;
static cudaStream_t stream = NULL;

static cusparseIndexBase_t index_base[2] = {CUSPARSE_INDEX_BASE_ZERO,
                                            CUSPARSE_INDEX_BASE_ONE};

struct cusparse_csr {
  cusparseMatDescr_t M;
  int *d_off, *d_col, *h_off, *h_col, *h_Q;
  double *d_val, *h_val;
};

static void csr_init(struct csr *A, const struct cholbench *cb) {
  struct cusparse_csr *B = tcalloc(struct cusparse_csr, 1);

  // Create desrciptor for M.
  chk_sparse(cusparseCreateMatDescr(&B->M));
  chk_sparse(cusparseSetMatIndexBase(B->M, index_base[A->base]));
  chk_sparse(cusparseSetMatType(B->M, CUSPARSE_MATRIX_TYPE_GENERAL));

  unsigned m = A->nrows, nnz = A->offs[m];
  // unsigned -> int since cusolver only likes ints.
  int *h_off = tcalloc(int, m + 1);
  for (unsigned i = 0; i < m + 1; i++)
    h_off[i] = A->offs[i] + A->base;

  int *h_col = tcalloc(int, nnz);
  for (unsigned i = 0; i < nnz; i++)
    h_col[i] = A->cols[i];

  // Find a reordering to minimize fill-in.
  B->h_Q = tcalloc(int, m);
  switch (cb->ordering) {
  case CHOLBENCH_ORDERING_RCM:
    chk_solver(
        cusolverSpXcsrsymrcmHost(solver, m, nnz, B->M, h_off, h_col, B->h_Q));
    break;
  case CHOLBENCH_ORDERING_AMD:
    chk_solver(
        cusolverSpXcsrsymamdHost(solver, m, nnz, B->M, h_off, h_col, B->h_Q));
    break;
  case CHOLBENCH_ORDERING_METIS:
    chk_solver(cusolverSpXcsrmetisndHost(solver, m, nnz, B->M, h_off, h_col,
                                         NULL, B->h_Q));
    break;
  case CHOLBENCH_ORDERING_NONE:
    for (unsigned i = 0; i < m; i++)
      B->h_Q[i] = i;
  default:
    break;
  }

  size_t bfr_size = 0;
  chk_solver(cusolverSpXcsrperm_bufferSizeHost(
      solver, m, m, nnz, B->M, h_off, h_col, B->h_Q, B->h_Q, &bfr_size));

  char *bfr = tcalloc(char, bfr_size);
  int *map = tcalloc(int, nnz);
  for (unsigned i = 0; i < nnz; i++)
    map[i] = i;
  chk_solver(cusolverSpXcsrpermHost(solver, m, m, nnz, B->M, h_off, h_col,
                                    B->h_Q, B->h_Q, map, (void *)bfr));
  tfree(bfr);

  // Reorder the matrix now.
  B->h_off = tcalloc(int, m + 1);
  for (unsigned i = 0; i < m + 1; i++)
    B->h_off[i] = h_off[i];
  chk_rt(cudaMalloc((void **)&B->d_off, (m + 1) * sizeof(int)));
  chk_rt(cudaMemcpy(B->d_off, B->h_off, (m + 1) * sizeof(int),
                    cudaMemcpyHostToDevice));
  tfree(h_off);

  B->h_col = tcalloc(int, nnz);
  for (unsigned i = 0; i < nnz; i++)
    B->h_col[i] = h_col[i];
  chk_rt(cudaMalloc((void **)&B->d_col, nnz * sizeof(int)));
  chk_rt(cudaMemcpy(B->d_col, B->h_col, nnz * sizeof(int),
                    cudaMemcpyHostToDevice));
  tfree(h_col);

  B->h_val = tcalloc(double, nnz);
  for (unsigned i = 0; i < nnz; i++)
    B->h_val[i] = A->vals[map[i]];
  chk_rt(cudaMalloc((void **)&B->d_val, nnz * sizeof(double)));
  chk_rt(cudaMemcpy(B->d_val, B->h_val, nnz * sizeof(double),
                    cudaMemcpyHostToDevice));
  tfree(map);

  A->ptr = (void *)B;
}

static void csr_finalize(struct csr *A) {
  struct cusparse_csr *B = (struct cusparse_csr *)A->ptr;
  if (B) {
    chk_sparse(cusparseDestroyMatDescr(B->M));
    chk_rt(cudaFree((void *)B->d_off));
    chk_rt(cudaFree((void *)B->d_col));
    chk_rt(cudaFree((void *)B->d_val));
    tfree(B->h_Q), tfree(B->h_off), tfree(B->h_col), tfree(B->h_val);
  }

  tfree(B), A->ptr = NULL;
}

int cusparse_init() {
  if (initialized)
    return 1;

  chk_rt(cudaStreamCreate(&stream));
  chk_solver(cusolverSpCreate(&solver));
  chk_sparse(cusparseCreate(&sparse));

  chk_solver(cusolverSpSetStream(solver, stream));
  chk_sparse(cusparseSetStream(sparse, stream));

  initialized = 1;

  return 0;
}

int cusparse_finalize() {
  if (initialized) {
    chk_solver(cusolverSpDestroy(solver));
    chk_sparse(cusparseDestroy(sparse));
    chk_rt(cudaStreamDestroy(stream));
    initialized = 0;
  }

  return 0;
}

void cusparse_bench(double *x, struct csr *A, const double *r,
                    const struct cholbench *cb) {
  csr_init(A, cb);

  unsigned m = A->nrows, nnz = A->offs[m];
  struct cusparse_csr *B = (struct cusparse_csr *)A->ptr;

  double *d_r, *d_x;
  chk_rt(cudaMalloc((void **)&d_r, m * sizeof(double)));
  chk_rt(cudaMalloc((void **)&d_x, m * sizeof(double)));

  double *tmp = tcalloc(double, m);
  for (unsigned i = 0; i < m; i++)
    tmp[i] = r[B->h_Q[i]];
  chk_rt(cudaMemcpy(d_r, tmp, m * sizeof(double), cudaMemcpyHostToDevice));

  // Warmup
  int singularity = 0;
  for (unsigned i = 0; i < cb->trials; i++) {
    chk_solver(cusolverSpDcsrlsvchol(solver, m, nnz, B->M, B->d_val, B->d_off,
                                     B->d_col, d_r, 1e-10, 0, d_x,
                                     &singularity));
  }

  // Time the solve
  chk_rt(cudaDeviceSynchronize());
  clock_t t = clock();
  for (unsigned i = 0; i < cb->trials; i++) {
    chk_solver(cusolverSpDcsrlsvchol(solver, m, nnz, B->M, B->d_val, B->d_off,
                                     B->d_col, d_r, 1e-10, 0, d_x,
                                     &singularity));
  }
  chk_rt(cudaDeviceSynchronize());
  t = clock() - t;

  chk_rt(cudaMemcpy(tmp, d_x, m * sizeof(double), cudaMemcpyDeviceToHost));
  chk_rt(cudaFree((void *)d_r));
  chk_rt(cudaFree((void *)d_x));

  for (unsigned i = 0; i < m; i++)
    x[B->h_Q[i]] = tmp[i];
  tfree(tmp);

  printf("===matrix,n,nnz,trials,solver,ordering,elapsed===\n");
  printf("%s,%u,%u,%u,%u,%d,%.15lf\n", cb->matrix, m, nnz, cb->trials,
         cb->solver, cb->ordering, (double)t / CLOCKS_PER_SEC);

  csr_finalize(A);
}

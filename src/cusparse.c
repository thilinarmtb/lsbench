#include "cholbench-impl.h"
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <err.h>
#include <stdio.h>

#define chk_solver(err)                                                        \
  {                                                                            \
    cusolverStatus_t err_ = (err);                                             \
    if (err_ != CUSOLVER_STATUS_SUCCESS)                                       \
      errx(EXIT_FAILURE, "%s:%d cusolverSp error: %s", __FILE__, __LINE__);    \
  }

#define chk_sparse(err)                                                        \
  {                                                                            \
    cusparseStatus_t err_ = (err);                                             \
    if (err_ != CUSPARSE_STATUS_SUCCESS)                                       \
      errx(EXIT_FAILURE, "%s:%d cusparse error: %s", __FILE__, __LINE__,       \
           cusparseGetErrorString(err_));                                      \
  }

#define chk_rt(err)                                                            \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess)                                                   \
      errx(EXIT_FAILURE, "%s:%d cuda error: %s", __FILE__, __LINE__,           \
           cudaGetErrorString(err_));                                          \
  }

static int initialized = 0;
static cusolverSpHandle_t solver = NULL;
static cusparseHandle_t sparse = NULL;
static cudaStream_t stream = NULL;

static cusparseIndexBase_t index_base[2] = {CUSPARSE_INDEX_BASE_ZERO,
                                            CUSPARSE_INDEX_BASE_ONE};

struct cusparse_csr {
  cusparseMatDescr_t M;
  int *d_off, *d_col;
  double *d_val;
};

static void csr_init(struct csr *A) {
  struct cusparse_csr *B = tcalloc(struct cusparse_csr, 1);

  // Create desrciptor for M.
  chk_sparse(cusparseCreateMatDescr(&B->M));
  chk_sparse(cusparseSetMatIndexBase(B->M, index_base[A->base]));
  chk_sparse(cusparseSetMatType(B->M, CUSPARSE_MATRIX_TYPE_GENERAL));

  unsigned m = A->nrows, nnz = A->offs[m];
  // unsigned -> int since cusolver only likes ints.
  int *offs = tcalloc(int, m + 1);
  for (unsigned i = 0; i < m + 1; i++)
    offs[i] = A->offs[i] + A->base;
  chk_rt(cudaMalloc((void **)&B->d_off, (m + 1) * sizeof(int)));
  chk_rt(cudaMemcpy(B->d_off, offs, (m + 1) * sizeof(int),
                    cudaMemcpyHostToDevice));
  tfree(offs);

  int *cols = tcalloc(int, nnz);
  for (unsigned i = 0; i < nnz; i++)
    cols[i] = A->cols[i];
  chk_rt(cudaMalloc((void **)&B->d_col, nnz * sizeof(int)));
  chk_rt(cudaMemcpy(B->d_col, cols, nnz * sizeof(int), cudaMemcpyHostToDevice));

  chk_rt(cudaMalloc((void **)&B->d_val, nnz * sizeof(double)));
  chk_rt(cudaMemcpy(B->d_val, A->vals, nnz * sizeof(double),
                    cudaMemcpyHostToDevice));
  tfree(cols);

  A->ptr = (void *)B;
}

static void csr_finalize(struct csr *A) {
  struct cusparse_csr *B = (struct cusparse_csr *)A->ptr;
  if (B) {
    chk_sparse(cusparseDestroyMatDescr(B->M));
    chk_rt(cudaFree(B->d_off));
    chk_rt(cudaFree(B->d_col));
    chk_rt(cudaFree(B->d_val));
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
                    unsigned ntrials) {
  csr_init(A);

  unsigned m = A->nrows, nnz = A->offs[m];
  double *d_r, *d_x;
  chk_rt(cudaMalloc((void **)&d_r, m * sizeof(double)));
  chk_rt(cudaMalloc((void **)&d_x, m * sizeof(double)));

  chk_rt(cudaMemcpy(d_r, r, m * sizeof(double), cudaMemcpyHostToDevice));

  struct cusparse_csr *B = (struct cusparse_csr *)A->ptr;
  int singularity = 0;
  for (unsigned i = 0; i < ntrials; i++) {
    chk_solver(cusolverSpDcsrlsvchol(solver, m, nnz, B->M, B->d_val, B->d_off,
                                     B->d_col, d_r, 1e-10, 0, d_x,
                                     &singularity));
    chk_rt(cudaDeviceSynchronize());
  }

  chk_rt(cudaMemcpy(x, d_x, m * sizeof(double), cudaMemcpyDeviceToHost));
  chk_rt(cudaFree(d_r));
  chk_rt(cudaFree(d_x));

  csr_finalize(A);
}

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
  cusparseMatDescr_t L;
  csrsv2Info_t info_L, info_Lt;
  cusparseSolvePolicy_t policy_L, policy_Lt;
  cusparseOperation_t trans_L, trans_Lt;
  int *d_offs, *d_cols;
  double *d_vals;
  void *d_bffr;
};

static void cusparse_csr_init(struct csr *A) {
  struct cusparse_csr *B = tcalloc(struct cusparse_csr, 1);

  // Create desrciptor for L.
  chk_sparse(cusparseCreateMatDescr(&B->L));
  chk_sparse(cusparseSetMatIndexBase(B->L, index_base[A->base]));
  chk_sparse(cusparseSetMatType(B->L, CUSPARSE_MATRIX_TYPE_GENERAL));
  chk_sparse(cusparseSetMatFillMode(B->L, CUSPARSE_FILL_MODE_LOWER));
  chk_sparse(cusparseSetMatDiagType(B->L, CUSPARSE_DIAG_TYPE_NON_UNIT));

  // Initialize policies and operations.
  B->policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  B->policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  B->trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
  B->trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

  unsigned m = A->nrows, nnz = A->offs[m];
  printf("m = %u, nnz = %u\n", m, nnz);

  // Copy matrix to device.
  chk_rt(cudaMalloc((void **)&B->d_offs, (m + 1) * sizeof(int)));
  int *offs = tcalloc(int, m + 1);
  for (unsigned i = 0; i < m + 1; i++)
    offs[i] = A->offs[i];
  chk_rt(cudaMemcpy(B->d_offs, offs, (m + 1) * sizeof(int),
                    cudaMemcpyHostToDevice));
  tfree(offs);

  chk_rt(cudaMalloc((void **)&B->d_cols, nnz * sizeof(int)));
  int *cols = tcalloc(int, nnz);
  for (unsigned i = 0; i < nnz; i++)
    cols[i] = A->cols[i];
  chk_rt(
      cudaMemcpy(B->d_cols, cols, nnz * sizeof(int), cudaMemcpyHostToDevice));
  tfree(cols);

  chk_rt(cudaMalloc((void **)&B->d_vals, nnz * sizeof(double)));
  chk_rt(cudaMemcpy(B->d_vals, A->vals, nnz * sizeof(double),
                    cudaMemcpyHostToDevice));

  // Create info structures.
  chk_sparse(cusparseCreateCsrsv2Info(&B->info_L));
  chk_sparse(cusparseCreateCsrsv2Info(&B->info_Lt));

  // Query memory requiremenets and allocate buffers.
  int bufsiz_L, bufsiz_Lt;
  chk_sparse(cusparseDcsrsv2_bufferSize(sparse, B->trans_L, m, nnz, B->L,
                                        B->d_vals, B->d_offs, B->d_cols,
                                        B->info_L, &bufsiz_L));
  chk_sparse(cusparseDcsrsv2_bufferSize(sparse, B->trans_Lt, m, nnz, B->L,
                                        B->d_vals, B->d_offs, B->d_cols,
                                        B->info_Lt, &bufsiz_Lt));

  int bufsiz = (bufsiz_L > bufsiz_Lt) ? bufsiz_L : bufsiz_Lt;
  chk_rt(cudaMalloc((void **)&B->d_bffr, bufsiz));

  chk_sparse(cusparseDcsrsv2_analysis(sparse, B->trans_L, m, nnz, B->L,
                                      B->d_vals, B->d_offs, B->d_cols,
                                      B->info_L, B->policy_L, B->d_bffr));
  chk_sparse(cusparseDcsrsv2_analysis(sparse, B->trans_Lt, m, nnz, B->L,
                                      B->d_vals, B->d_offs, B->d_cols,
                                      B->info_Lt, B->policy_Lt, B->d_bffr));
}

static void cusparse_csr_finalize(struct csr *A) {
  struct cusparse_csr *B = (struct cusparse_csr *)A->ptr;
  if (B) {
    chk_sparse(cusparseDestroyMatDescr(B->L));
    chk_sparse(cusparseDestroyCsrsv2Info(B->info_L));
    chk_sparse(cusparseDestroyCsrsv2Info(B->info_Lt));
    chk_rt(cudaFree(B->d_offs));
    chk_rt(cudaFree(B->d_cols));
    chk_rt(cudaFree(B->d_vals));
    chk_rt(cudaFree(B->d_bffr));
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
  cusparse_csr_init(A);
  unsigned m = A->nrows, nnz = A->offs[m];
  printf("m = %u, nnz = %u\n", m, nnz);

  double *d_r, *d_z, *d_x;
  chk_rt(cudaMalloc((void **)&d_r, m * sizeof(double)));
  chk_rt(cudaMalloc((void **)&d_z, m * sizeof(double)));
  chk_rt(cudaMalloc((void **)&d_x, m * sizeof(double)));

  chk_rt(cudaMemcpy(d_r, r, m * sizeof(double), cudaMemcpyHostToDevice));

  const double alpha = 1.0;
  struct cusparse_csr *B = (struct cusparse_csr *)A->ptr;
  for (unsigned i = 0; i < ntrials; i++) {
    chk_sparse(cusparseDcsrsv2_solve(sparse, B->trans_L, m, nnz, &alpha, B->L,
                                     B->d_vals, B->d_offs, B->d_cols, B->info_L,
                                     d_r, d_z, B->policy_L, B->d_bffr));
    // chk_sparse(cusparseDcsrsv2_solve(sparse, B->trans_Lt, m, nnz, &alpha,
    // B->L,
    //                                B->d_vals, B->d_offs, B->d_cols,
    //                                B->info_Lt, d_z, d_x, B->policy_Lt,
    //                                B->d_bffr));
  }

  chk_rt(cudaMemcpy(d_x, x, m * sizeof(double), cudaMemcpyDeviceToHost));

  chk_rt(cudaFree(d_z));
  chk_rt(cudaFree(d_r));
  chk_rt(cudaFree(d_x));

  cusparse_csr_finalize(A);
}

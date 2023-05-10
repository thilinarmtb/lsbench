#include "lsbench-impl.h"
#include <math.h>

#if defined(LSBENCH_HYPRE)
#define NPARAM 11
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <_hypre_utilities.h>

static int initialized = 0;
static HYPRE_Solver solver = 0;

struct hypre_csr {
  HYPRE_IJMatrix A;
  HYPRE_IJVector b, x;
};

static void csr_finalize(struct hypre_csr *A) {
  if (A) {
    HYPRE_IJMatrixDestroy(A->A);
    HYPRE_IJVectorDestroy(A->x);
    HYPRE_IJVectorDestroy(A->b);
  }
  tfree(A);
}

int hypre_init() {
  if (initialized)
    return 1;

  HYPRE_Init();
  HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

  // Settings for cuda
  HYPRE_Int spgemm_use_vendor = 0;
  HYPRE_SetSpGemmUseVendor(spgemm_use_vendor);
  HYPRE_SetSpMVUseVendor(1);

  HYPRE_SetUseGpuRand(1);

  double params[NPARAM];
  params[0] = 8;    /* coarsening */
  params[1] = 6;    /* interpolation */
  params[2] = 2;    /* number of cycles */
  params[3] = 8;    /* smoother for crs level */
  params[4] = 3;    /* sweeps */
  params[5] = 8;    /* smoother */
  params[6] = 1;    /* sweeps   */
  params[7] = 0.25; /* threshold */
  params[8] = 0.00; /* non galerkin tolerance */
  params[9] = 0;    /* agressive coarsening */
  params[10] = 2;   /* chebyRelaxOrder */

  HYPRE_BoomerAMGCreate(&solver);

  HYPRE_BoomerAMGSetCoarsenType(solver, params[0]);
  HYPRE_BoomerAMGSetInterpType(solver, params[1]);

  HYPRE_BoomerAMGSetModuleRAP2(solver, 1);
  HYPRE_BoomerAMGSetKeepTranspose(solver, 1);

  // HYPRE_BoomerAMGSetChebyOrder(solver, params[10]);
  // HYPRE_BoomerAMGSetChebyFraction(*solver, 0.2);

  if (params[5] > 0) {
    HYPRE_BoomerAMGSetCycleRelaxType(solver, params[5], 1);
    HYPRE_BoomerAMGSetCycleRelaxType(solver, params[5], 2);
  }
  HYPRE_BoomerAMGSetCycleRelaxType(solver, 9, 3);

  HYPRE_BoomerAMGSetCycleNumSweeps(solver, params[6], 1);
  HYPRE_BoomerAMGSetCycleNumSweeps(solver, params[6], 2);
  HYPRE_BoomerAMGSetCycleNumSweeps(solver, 1, 3);

  int null_space = 0;
  if (null_space) {
    HYPRE_BoomerAMGSetMinCoarseSize(solver, 2);
    HYPRE_BoomerAMGSetCycleRelaxType(solver, params[3], 3);
    HYPRE_BoomerAMGSetCycleNumSweeps(solver, params[4], 3);
  }

  HYPRE_BoomerAMGSetStrongThreshold(solver, params[7]);

// NonGalerkin not supported yet
#if 0
  if (params[8] > 1e-3) {
    HYPRE_BoomerAMGSetNonGalerkinTol(*solver,params[8]);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(*solver,0.0 , 0);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(*solver,0.01, 1);
    HYPRE_BoomerAMGSetLevelNonGalerkinTol(*solver,0.05, 2);
  }
#endif

  if (params[9] > 0) {
    HYPRE_BoomerAMGSetAggNumLevels(solver, params[9]);
    HYPRE_BoomerAMGSetAggInterpType(solver, 5);
    // HYPRE_BoomerAMGSetNumPaths(solver, 10);
  }

  HYPRE_BoomerAMGSetMaxIter(solver, params[2]);
  HYPRE_BoomerAMGSetTol(solver, 0);

  //  HYPRE_BoomerAMGSetPrintLevel(solver, 3);

  initialized = 1;
  return 0;
}

int hypre_finalize() {
  if (!initialized)
    return 1;

  HYPRE_BoomerAMGDestroy(solver);
  HYPRE_Finalize();
  initialized = 0;
  return 0;
}

#if defined(HYPRE_CUDA_ENABLED)
#include <cuda_runtime.h>

#define chk_rt(err)                                                            \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      errx(EXIT_FAILURE, "%s:%d cuda error: %s", __FILE__, __LINE__,           \
           cudaGetErrorString(err_));                                          \
    }                                                                          \
  }

static struct hypre_csr *csr_init(struct csr *A, const struct lsbench *cb) {
  struct hypre_csr *B = tcalloc(struct hypre_csr, 1);

  int comm = 0;
  HYPRE_BigInt lower = A->base, upper = (HYPRE_BigInt)A->nrows - 1 + A->base;
  HYPRE_IJMatrixCreate(comm, lower, upper, lower, upper, &B->A);
  HYPRE_IJMatrixSetObjectType(B->A, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(B->A);

  unsigned nr = A->nrows, nnz = A->offs[nr];
  HYPRE_BigInt *rows = tcalloc(HYPRE_BigInt, nr);
  HYPRE_BigInt *cols = tcalloc(HYPRE_BigInt, nnz);
  HYPRE_Int *ncols = tcalloc(HYPRE_Int, nr);
  HYPRE_Real *vals = tcalloc(HYPRE_Real, nnz);
  for (unsigned r = 0; r < nr; r++) {
    rows[r] = (HYPRE_BigInt)(r + A->base);
    for (unsigned j = A->offs[r], je = A->offs[r + 1]; j < je; j++) {
      cols[j] = (HYPRE_BigInt)A->cols[j];
      vals[j] = (HYPRE_Real)A->vals[j];
    }
    ncols[r] = (HYPRE_Int)(A->offs[r + 1] - A->offs[r]);
  }

  HYPRE_BigInt *d_rows, *d_cols;
  chk_rt(cudaMalloc((void **)&d_rows, nr * sizeof(HYPRE_BigInt)));
  chk_rt(cudaMalloc((void **)&d_cols, nnz * sizeof(HYPRE_BigInt)));
  chk_rt(cudaMemcpy(d_rows, rows, nr * sizeof(HYPRE_BigInt),
                    cudaMemcpyHostToDevice));
  chk_rt(cudaMemcpy(d_cols, cols, nnz * sizeof(HYPRE_BigInt),
                    cudaMemcpyHostToDevice));
  tfree(rows), tfree(cols);

  HYPRE_Int *d_ncols;
  chk_rt(cudaMalloc((void **)&d_ncols, nr * sizeof(HYPRE_Int)));
  chk_rt(cudaMemcpy(d_ncols, ncols, nr * sizeof(HYPRE_Int),
                    cudaMemcpyHostToDevice));
  tfree(ncols);

  HYPRE_Real *d_vals;
  chk_rt(cudaMalloc((void **)&d_vals, nnz * sizeof(HYPRE_Real)));
  chk_rt(cudaMemcpy(d_vals, vals, nnz * sizeof(HYPRE_Real),
                    cudaMemcpyHostToDevice));
  tfree(vals);

  HYPRE_IJMatrixSetValues(B->A, nr, d_ncols, d_rows, d_cols, d_vals);
  HYPRE_IJMatrixAssemble(B->A);
  // HYPRE_IJMatrixPrint(B->A, "A.dat");

  // chk_rt(cudaFree((void *)d_rows));
  // chk_rt(cudaFree((void *)d_cols));
  // chk_rt(cudaFree((void *)d_ncols));
  // chk_rt(cudaFree((void *)d_vals));

  // Create and initialize rhs and solution vectors
  HYPRE_IJVectorCreate(comm, lower, upper, &B->b);
  HYPRE_IJVectorSetObjectType(B->b, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(B->b);
  HYPRE_IJVectorAssemble(B->b);

  HYPRE_IJVectorCreate(comm, lower, upper, &B->x);
  HYPRE_IJVectorSetObjectType(B->x, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(B->x);
  HYPRE_IJVectorAssemble(B->x);

  HYPRE_ParVector par_b, par_x;
  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  if (cb->verbose == 1) {
    HYPRE_BoomerAMGSetPrintLevel(solver, 1);
  } else if (cb->verbose > 1) {
    HYPRE_BoomerAMGSetPrintLevel(solver, 3);
  }
  HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);

  return B;
}

int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  if (!initialized)
    return 1;

  struct hypre_csr *B = csr_init(A, cb);

  unsigned nr = A->nrows, nnz = A->offs[nr];
  HYPRE_Real *d_r, *d_x;
  chk_rt(cudaMalloc((void **)&d_x, nr * sizeof(HYPRE_Real)));
  chk_rt(cudaMalloc((void **)&d_r, nr * sizeof(HYPRE_Real)));

  HYPRE_Real *tmp = tcalloc(HYPRE_Real, nr);
  for (unsigned i = 0; i < nr; i++)
    tmp[i] = r[i];
  chk_rt(cudaMemcpy(d_r, tmp, nr * sizeof(HYPRE_Real), cudaMemcpyHostToDevice));

  HYPRE_IJVectorUpdateValues(B->x, nr, NULL, d_x, 1);
  HYPRE_IJVectorUpdateValues(B->b, nr, NULL, d_r, 1);

  HYPRE_ParVector par_x, par_b;
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);

  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  // HYPRE_IJVectorPrint(B->b, "b.dat");
  // HYPRE_IJVectorPrint(B->x, "x.dat");

  // Warmup
  HYPRE_BoomerAMGSetPrintLevel(solver, 0);
  for (unsigned i = 0; i < cb->trials; i++)
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);

  if (cb->verbose > 1) {
    HYPRE_BoomerAMGSetPrintLevel(solver, 3);
  }
  // Time the solve
  chk_rt(cudaDeviceSynchronize());
  clock_t t = clock();
  for (unsigned i = 0; i < cb->trials; i++)
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
  chk_rt(cudaDeviceSynchronize());
  t = clock() - t;
  // FIXME: Print the solve times.

  HYPRE_IJVectorGetValues(B->x, nr, NULL, d_x);
  chk_rt(cudaMemcpy(tmp, d_x, nr * sizeof(HYPRE_Real), cudaMemcpyDeviceToHost));
  chk_rt(cudaFree((void *)d_r));
  chk_rt(cudaFree((void *)d_x));

  for (unsigned i = 0; i < nr; i++)
    x[i] = tmp[i];

  csr_finalize(B), tfree(tmp);
  return 0;
}

#undef chk_rt
#elif defined(HYPRE_HIP_ENABLED) // HIP
#include <hip_runtime.h>

#define chk_rt(err)                                                            \
  {                                                                            \
    hipError_t err_ = (err);                                                   \
    if (err_ != hipSuccess) {                                                  \
      errx(EXIT_FAILURE, "%s:%d hip error: %s", __FILE__, __LINE__,            \
           hipGetErrorString(err_));                                           \
    }                                                                          \
  }

int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  return 1;
}
#undef chk_rt
#elif defined(HYPRE_DPCPP_ENABLED) // DPCPP
int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  return 1;
}

#undef chk_rt
#else // HYPRE CPU
/*
int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  return 1;
}
*/
static struct hypre_csr *csr_init(struct csr *A, const struct lsbench *cb) {
  struct hypre_csr *B = tcalloc(struct hypre_csr, 1);

  int comm = 0;
  HYPRE_BigInt lower = A->base, upper = (HYPRE_BigInt)A->nrows - 1 + A->base;
  HYPRE_IJMatrixCreate(comm, lower, upper, lower, upper, &B->A);
  HYPRE_IJMatrixSetObjectType(B->A, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(B->A);

  unsigned nr = A->nrows, nnz = A->offs[nr];
  HYPRE_BigInt *rows = tcalloc(HYPRE_BigInt, nr);
  HYPRE_BigInt *cols = tcalloc(HYPRE_BigInt, nnz);
  HYPRE_Int *ncols = tcalloc(HYPRE_Int, nr);
  HYPRE_Real *vals = tcalloc(HYPRE_Real, nnz);
  for (unsigned r = 0; r < nr; r++) {
    rows[r] = (HYPRE_BigInt)(r + A->base);
    for (unsigned j = A->offs[r], je = A->offs[r + 1]; j < je; j++) {
      cols[j] = (HYPRE_BigInt)A->cols[j];
      vals[j] = (HYPRE_Real)A->vals[j];
    }
    ncols[r] = (HYPRE_Int)(A->offs[r + 1] - A->offs[r]);
  }

  HYPRE_IJMatrixSetValues(B->A, nr, ncols, rows, cols, vals);
  HYPRE_IJMatrixAssemble(B->A);
  // HYPRE_IJMatrixPrint(B->A, "A.dat");

  tfree(rows), tfree(cols);
  tfree(ncols);
  tfree(vals);

  // Create and initialize rhs and solution vectors
  HYPRE_IJVectorCreate(comm, lower, upper, &B->b);
  HYPRE_IJVectorSetObjectType(B->b, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(B->b);
  HYPRE_IJVectorAssemble(B->b);

  HYPRE_IJVectorCreate(comm, lower, upper, &B->x);
  HYPRE_IJVectorSetObjectType(B->x, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(B->x);
  HYPRE_IJVectorAssemble(B->x);

  HYPRE_ParVector par_b, par_x;
  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  if (cb->verbose == 1) {
    HYPRE_BoomerAMGSetPrintLevel(solver, 1);
  } else if (cb->verbose > 1) {
    HYPRE_BoomerAMGSetPrintLevel(solver, 3);
  }
  HYPRE_BoomerAMGSetup(solver, par_A, par_b, par_x);

  return B;
}

int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  if (!initialized)
    return 1;

  struct hypre_csr *B = csr_init(A, cb);

  unsigned nr = A->nrows, nnz = A->offs[nr];

  HYPRE_IJVectorUpdateValues(B->x, nr, NULL, (HYPRE_Real *)x, 1);
  HYPRE_IJVectorUpdateValues(B->b, nr, NULL, (HYPRE_Real *)r, 1);

  HYPRE_ParVector par_x, par_b;
  HYPRE_IJVectorGetObject(B->x, (void **)&par_x);
  HYPRE_IJVectorGetObject(B->b, (void **)&par_b);

  HYPRE_ParCSRMatrix par_A;
  HYPRE_IJMatrixGetObject(B->A, (void **)&par_A);

  // HYPRE_IJVectorPrint(B->b, "b.dat");
  // HYPRE_IJVectorPrint(B->x, "x.dat");

  // Warmup

  if (cb->verbose > 1) {
    HYPRE_BoomerAMGSetPrintLevel(solver, 2);
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
  }

  HYPRE_BoomerAMGSetPrintLevel(solver, 0);
  for (unsigned i = 0; i < cb->trials; i++)
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);

  // Time the solve
  clock_t t = clock();
  for (unsigned i = 0; i < cb->trials; i++)
    HYPRE_BoomerAMGSolve(solver, par_A, par_b, par_x);
  t = clock() - t;
  // FIXME: Print the solve times.

  HYPRE_IJVectorGetValues(B->x, nr, NULL, (HYPRE_Real *)x);

  /*
    if (cb->verbose > 0) {
      HYPRE_IJVector rd;
      HYPRE_IJVectorUpdateValues(rd, nr, NULL, (HYPRE_Real *)B->b, 1);

      HYPRE_Real norm;
      HYPRE_StructMatrixMatvec((HYPRE_Real) -1.0, B->A, B->x, (HYPRE_Real) 1.0,
    B->b); HYPRE_IJVectorInnerProd(rd, rd, &norm);

      printf("norm(b-Ax) = %e\n", sqrt(norm));
      HYPRE_IJVectorDestroy(rd);
    }
  */
  /*
    for (int i=0;i<nr;i++){
      printf("dbg x  %d   %e\n",i,x[i]);
    }
  */
  //  printf("dbg x  %e %e %e\n",glmin(x,nr), glmax(x,nr), glamax(x,nr));

  csr_finalize(B);
  return 0;
}
#endif
#undef NPARAM
#else // LSBENCH_HYPRE
int hypre_init() { return 1; }
int hypre_finalize() { return 1; }
int hypre_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  return 1;
}
#endif

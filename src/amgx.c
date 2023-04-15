#include "lsbench-impl.h"
#include <amgx_c.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <err.h>
//#include <mpi.h>
#include <stdio.h>
#include <time.h>

#define chk_rt(err)                                                            \
  {                                                                            \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      errx(EXIT_FAILURE, "%s:%d cuda error: %s", __FILE__, __LINE__,           \
           cudaGetErrorString(err_));                                          \
    }                                                                          \
  }

static int initialized = 0;
static AMGX_solver_handle solver = 0;
static AMGX_config_handle config = 0;
static AMGX_resources_handle resource = 0;
static AMGX_Mode mode;

struct amgx_csr {
  AMGX_vector_handle x, b;
  AMGX_matrix_handle A;
};

static void csr_init(struct csr *A, const struct lsbench *cb) {
  struct amgx_csr *B = tcalloc(struct amgx_csr, 1);

  AMGX_SAFE_CALL(AMGX_vector_create(&B->x, resource, mode));
  AMGX_SAFE_CALL(AMGX_vector_create(&B->b, resource, mode));
  AMGX_SAFE_CALL(AMGX_matrix_create(&B->A, resource, mode));

  unsigned nr = A->nrows, nnz = A->offs[nr];
  int *offs = tcalloc(int, nr + 1);
  for (unsigned i = 0; i < nr + 1; i++)
    offs[i] = A->offs[i];

  int *cols = tcalloc(int, nnz);
  for (unsigned i = 0; i < nnz; i++)
    cols[i] = A->cols[i] - A->base;

  float *vals = tcalloc(float, nnz);
  for (unsigned i = 0; i < nnz; i++)
    vals[i] = A->vals[i];

  AMGX_SAFE_CALL(
      AMGX_matrix_upload_all(B->A, nr, nnz, 1, 1, offs, cols, vals, NULL));
  tfree(vals), tfree(offs), tfree(cols);

  AMGX_SAFE_CALL(AMGX_solver_setup(solver, B->A));
  AMGX_SAFE_CALL(AMGX_vector_bind(B->x, B->A));
  AMGX_SAFE_CALL(AMGX_vector_bind(B->b, B->A));

  A->ptr = (void *)B;
}

static void csr_finalize(struct csr *A) {
  struct amgx_csr *B = (struct amgx_csr *)A->ptr;
  if (B) {
    AMGX_SAFE_CALL(AMGX_vector_destroy(B->x));
    AMGX_SAFE_CALL(AMGX_vector_destroy(B->b));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(B->A));
  }
  tfree(B), A->ptr = NULL;
}

void amgx_print(const char *msg, int length) { printf("%s", msg); }

int amgx_init() {
  if (initialized)
    return 1;

  AMGX_SAFE_CALL(AMGX_initialize());
  AMGX_SAFE_CALL(AMGX_initialize_plugins());
  AMGX_SAFE_CALL(AMGX_register_print_callback(&amgx_print));
  AMGX_SAFE_CALL(AMGX_install_signal_handler());

  char *cfg = "{\"config_version\":2,\"solver\":{\"scope\":\"main\",\"solver\":"
              "\"AMG\",\"algorithm\":"
              "\"CLASSICAL\",\"strength_threshold\":0.25,\"max_row_sum\":0.9,"
              "\"interpolator\":\"D2\",\"interp_max_elements\":4,\"max_"
              "levels\":20,\"print_config\":0,\"print_grid_stats\":1,\"max_"
              "iters\":1,\"cycle\":\"V\",\"presweeps\":1,\"postsweeps\":1, "
              "\"coarsest_sweeps\":3,\"use_sum_stopping_criteria\":1, "
              "\"coarse_solver\": \"NOSOLVER\"}}";
  AMGX_SAFE_CALL(AMGX_config_create(&config, cfg));

  int device = 0;
  AMGX_SAFE_CALL(AMGX_resources_create(&resource, config, NULL, 1, &device));

  mode = AMGX_mode_dFFI;
  AMGX_SAFE_CALL(AMGX_solver_create(&solver, resource, mode, config));

  initialized = 1;

  return 0;
}

void amgx_bench(double *x, struct csr *A, const double *r,
                const struct lsbench *cb) {
  csr_init(A, cb);

  unsigned nr = A->nrows;
  float *rf = tcalloc(float, nr);
  for (unsigned i = 0; i < nr; i++)
    rf[i] = r[i];
  float *xf = tcalloc(float, nr);

  struct amgx_csr *B = (struct amgx_csr *)A->ptr;
  AMGX_SAFE_CALL(AMGX_vector_upload(B->b, nr, 1, rf));
  AMGX_SAFE_CALL(AMGX_vector_upload(B->x, nr, 1, xf));

  // Warmup
  for (unsigned i = 0; i < cb->trials; i++)
    AMGX_solver_solve(solver, B->b, B->x);

  // Time the solve
  chk_rt(cudaDeviceSynchronize());
  clock_t t = clock();
  for (unsigned i = 0; i < cb->trials; i++)
    AMGX_solver_solve(solver, B->b, B->x);
  chk_rt(cudaDeviceSynchronize());
  t = clock() - t;

  AMGX_SAFE_CALL(AMGX_vector_download(B->x, xf));
  for (unsigned i = 0; i < nr; i++)
    x[i] = xf[i];

  printf("x =\n");
  for (unsigned i = 0; i < nr; i++)
    printf("%lf\n", x[i]);

  csr_finalize(A), tfree(rf), tfree(xf);
}

int amgx_finalize() {
  if (initialized) {
    AMGX_solver_destroy(solver);
    AMGX_resources_destroy(resource);
    AMGX_SAFE_CALL(AMGX_config_destroy(config));
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());
    initialized = 0;
  }

  return 0;
}

#include "assert.h"
#include "lsbench-impl.h"

#if defined(LSBENCH_PARALMOND)
#include "parAlmond.hpp"
#include "platform.hpp"

static int initialized = 0;
static libp::comm_t *comm = NULL;
static libp::platformSettings_t *set_plat = NULL;
static libp::platform_t *platform = NULL;
static libp::settings_t *set_amg = NULL;

struct paralmond_csr {
  libp::parAlmond::parAlmond_t *parAlmond;
  libp::deviceMemory<double> *d_b, *d_x;
  uint nr;
};

static void csr_init(struct csr *A, const struct lsbench *cb) {
  struct paralmond_csr *B = tcalloc(struct paralmond_csr, 1);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for (uint r = 0; r < size; r++) {
    if (r == rank) {
      B->parAlmond =
          new libp::parAlmond::parAlmond_t(*platform, *set_amg, *comm);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Translate our CSR to parCOO
  libp::parAlmond::parCOO C(*platform, *comm);

  // C.globalRowStarts, C.globalColStarts
  assert(comm->size() == 1);
  C.globalRowStarts.malloc(2, 0);
  C.globalColStarts.malloc(2, 0);
  C.globalRowStarts[0] = 0, C.globalRowStarts[1] = A->nrows;
  C.globalColStarts[0] = 0, C.globalColStarts[1] = A->nrows;

  // C.nnz, C.entries,
  C.nnz = A->offs[A->nrows];
  C.entries.malloc(C.nnz);
  for (uint r = 0; r < A->nrows; r++) {
    for (uint j = A->offs[r]; j < A->offs[r + 1]; j++) {
      C.entries[j].row = r;
      C.entries[j].col = A->cols[j] - A->base;
      C.entries[j].val = A->vals[j];
    }
  }

  libp::memory<dfloat> null(A->nrows);
  for (uint i = 0; i < A->nrows; i++)
    null[i] = 1.0 / sqrt(A->nrows);

  uint null_space = 0;
  for (uint r = 0; r < size; r++) {
    if (r == rank) {
      B->parAlmond->AMGSetup(C, null_space, null, 1.0);
      B->parAlmond->Report();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  uint nr = B->parAlmond->getNumRows(0);
  uint nc = B->parAlmond->getNumCols(0);
  assert(nr == nc);
  B->nr = A->nrows;

  occa::device device = platform->device;
  B->d_b = new libp::deviceMemory<double>(device.malloc<double>(A->nrows));
  B->d_x = new libp::deviceMemory<double>(device.malloc<double>(A->nrows));

  A->ptr = (void *)B;
}

int paralmond_init() {
  if (initialized)
    return 0;

  // Setup the libParanumal global MPI communicator.
  MPI_Init(NULL, NULL);
  libp::Comm::Init(MPI_COMM_WORLD);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  comm = new libp::comm_t[1];
  *comm = libp::Comm::World().Dup().Split(rank, rank);

  // Setup the settings for platform settings.
  set_plat = new libp::platformSettings_t(*comm);
  set_plat->report();
  platform = new libp::platform_t(*set_plat);

  // Setup the settings for parAlmond.
  set_amg = new libp::settings_t(*comm);
  libp::parAlmond::AddSettings(*set_amg, "");
  set_amg->report();

  initialized = 1;
  return 0;
}

void paralmond_bench(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb) {
  csr_init(A, cb);

  struct paralmond_csr *B = (struct paralmond_csr *)A->ptr;

  libp::memory<double> h_r = libp::memory<double>(B->nr);
  for (uint i = 0; i < B->nr; i++)
    h_r[i] = r[i];
  B->d_b->copyFrom(h_r);

  B->parAlmond->Operator(*(B->d_b), *(B->d_x));

  libp::memory<double> h_x = libp::memory<double>(B->nr);
  B->d_x->copyTo(h_x);

  double *xx = h_x.ptr();
  for (unsigned i = 0; i < B->nr; i++)
    x[i] = xx[i];

  delete B->d_b, B->d_x;
  delete B->parAlmond;
  free(B);
}

int paralmond_finalize() {
  if (initialized) {
    delete[] comm;
    delete set_plat, set_amg, platform;
    initialized = 0;
  }
  return 0;
}

#else
int paralmond_init();
int paralmond_finalize();
void paralmond_bench(double *x, struct csr *A, const double *r,
                     const struct lsbench *cb);
#endif

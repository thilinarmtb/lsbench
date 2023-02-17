#include "cholbench-impl.h"
#include <assert.h>
#include <err.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cholbench *cholbench_init(int argc, char *argv[]) {
  struct option long_options[] = {{"matrix", required_argument, 0, 0},
                                  {"solver", optional_argument, 0, 1},
                                  {"ordering", optional_argument, 0, 2},
                                  {"verbose", optional_argument, 0, 3},
                                  {"trials", optional_argument, 0, 4},
                                  {0, 0, 0, 0}};

  // Create the struct and set defauls.
  struct cholbench *cb = tcalloc(struct cholbench, 1);
  cb->solver = CHOLBENCH_SOLVER_CUSOLVER;
  cb->ordering = CHOLBENCH_ORDERING_NONE;
  cb->verbose = 0;
  cb->trials = 50;

  // Parse the command line arguments.
  for (;;) {
    int idx = 0;
    int c = getopt_long(argc, argv, "", long_options, &idx);
    if (c == -1)
      break;

    size_t len;
    switch (c) {
    case 0:
      len = strnlen(optarg, BUFSIZ);
      cb->matrix = tcalloc(char, len + 1);
      strncpy(cb->matrix, optarg, len);
      break;
    case 1:
      cb->solver = atoi(optarg);
      break;
    case 2:
      cb->ordering = atoi(optarg);
      break;
    case 3:
      cb->verbose = atoi(optarg);
      break;
    case 4:
      cb->trials = atoi(optarg);
      break;
    default:
      errx(EXIT_FAILURE, "Unknown command line option.");
      break;
    }
  }

  cusparse_init();

  return cb;
}

struct coo_entry {
  unsigned r, c;
  double v;
};

static int cmp_coo(const void *va, const void *vb) {
  struct coo_entry *a = (struct coo_entry *)va, *b = (struct coo_entry *)vb;

  // Compare rows first.
  if (a->r < b->r)
    return -1;
  else if (a->r > b->r)
    return 1;
  // a->r == b->r, so we check columns.
  else if (a->c < b->c)
    return -1;
  else if (a->c > b->c)
    return 1;
  // Entries are the same.
  return 0;
}

struct csr *cholbench_matrix_read(const struct cholbench *cb) {
  FILE *fp = fopen(cb->matrix, "r");
  if (!fp)
    err(EXIT_FAILURE, "Unable to open file \"%s\" for reading", cb->matrix);

  // Read total number of non-zero entries.
  unsigned nnz, base;
  char ch;
  int ret = fscanf(fp, "%u %u%c", &nnz, &base, &ch);
  if (ret != 3 || (ch != '\n' && ch != EOF))
    errx(EXIT_FAILURE, "Unable to read meta information about the matrix.");
  if (base > 1)
    err(EXIT_FAILURE, "Base should be either 0 or 1, got: %u.\n", base);
  if (nnz == 0)
    return NULL;

  struct coo_entry *arr = tcalloc(struct coo_entry, nnz);
  if (arr == NULL)
    err(EXIT_FAILURE, "Unable to allocate memories for %u COO entries", nnz);

  for (unsigned i = 0; i < nnz; i++) {
    ret = fscanf(fp, "%u %u %lf%c", &arr[i].r, &arr[i].c, &arr[i].v, &ch);
    if (ret != 4 || (ch != '\n' && ch != EOF))
      errx(EXIT_FAILURE, "Unable to read matrix entries.");
  }
  qsort(arr, nnz, sizeof(struct coo_entry), cmp_coo);

  // Sum up the repeated entries if there are any and compress the array.
  unsigned nnzc = 0, s = 0, e;
  while (s < nnz) {
    arr[nnzc] = arr[s], e = s + 1;
    while (arr[s].r == arr[e].r && arr[s].c == arr[e].c)
      arr[nnzc].v += arr[e].v, e++;
    s = e, nnzc++;
  }

  // Count the number of rows.
  unsigned nrows = 1;
  for (unsigned i = 1; i < nnzc; i++) {
    if (arr[i - 1].r != arr[i].r)
      nrows++;
  }

  // Allocate arrays to hold the entries in CSR format.
  struct csr *A = tcalloc(struct csr, 1);
  A->nrows = nrows, A->base = base;
  A->offs = tcalloc(unsigned, nrows + 1);
  A->cols = tcalloc(unsigned, nnzc);
  A->vals = tcalloc(double, nnzc);

  unsigned crows = 1;
  A->offs[0] = 0, A->cols[0] = arr[0].c, A->vals[0] = arr[0].v;
  for (unsigned i = 1; i < nnzc; i++) {
    if (arr[i - 1].r != arr[i].r)
      A->offs[crows] = i, crows++;
    A->cols[i] = arr[i].c, A->vals[i] = arr[i].v;
  }
  A->offs[crows] = nnzc;
  // Sanity check.
  assert(crows == nrows);

  tfree(arr), fclose(fp);
  return A;
}

void cholbench_bench(struct csr *A, const struct cholbench *cb) {
  unsigned m = A->nrows;
  double *x = tcalloc(double, m), *r = tcalloc(double, m);
  for (unsigned i = 0; i < m; i++)
    r[i] = i;

  switch (cb->solver) {
  case 0:
    cusparse_bench(x, A, r, cb);
    break;
  default:
    errx(EXIT_FAILURE, "Unknown solver: %d.", cb->solver);
    break;
  }

  tfree(x), tfree(r);
}

void cholbench_matrix_print(const struct csr *A) {
  for (unsigned i = 0; i < A->nrows; i++) {
    for (unsigned s = A->offs[i], e = A->offs[i + 1]; s < e; s++)
      printf("%u %u %lf\n", i + A->base, A->cols[s], A->vals[s]);
  }
}

void cholbench_matrix_free(struct csr *A) {
  if (A) {
    tfree(A->offs);
    tfree(A->cols);
    tfree(A->vals);
  }
  tfree(A);
}

void cholbench_finalize(struct cholbench *cb) {
  cusparse_finalize();
  if (cb)
    tfree(cb->matrix);
  tfree(cb);
}

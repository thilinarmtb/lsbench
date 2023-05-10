#include "lsbench-impl.h"
#include <assert.h>
#include <err.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

struct csr *lsbench_matrix_read(const char *fname) {
  FILE *fp = fopen(fname, "r");
  if (!fp)
    err(EXIT_FAILURE, "Unable to open file \"%s\" for reading", fname);

  // Read total number of non-zero entries.
  unsigned nnz, base;
  char ch;
  int ret = fscanf(fp, "%u %u%c", &nnz, &base, &ch);
  if (ret != 3 || (ch != '\n' && ch != EOF))
    errx(EXIT_FAILURE, "Unable to read meta information about the matrix.");
  if (base > 1)
    err(EXIT_FAILURE, "Base should be either 0 or 1, got: %u.\n", base);
  if (nnz == 0)
    err(EXIT_FAILURE, "Number of nnz values in the file are zero.");

  struct coo_entry *arr = tcalloc(struct coo_entry, nnz);
  if (arr == NULL)
    err(EXIT_FAILURE, "Unable to allocate memories for %u COO entries.", nnz);

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
    while (e < nnz && arr[s].r == arr[e].r && arr[s].c == arr[e].c)
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

  csr_symA(A);

  tfree(arr), fclose(fp);
  return A;
}

// A = (A + A.T)/2;
void csr_symA(struct csr *A) {
  for (unsigned i = 0; i < A->nrows; i++) {
    for (unsigned j = A->offs[i]; j < A->offs[i + 1]; j++) {
      if (A->cols[j] - A->base <= i) {
        unsigned irow = A->cols[j] - A->base;
        for (unsigned k = A->offs[irow]; k < A->offs[irow + 1]; k++) {
          if (A->cols[k] == i + A->base) {
            double tmp = 0.5 * (A->vals[j] + A->vals[k]);
            A->vals[j] = tmp;
            A->vals[k] = tmp;
          }
        }
      }
    }
  }
}

// y = a A*x + b y
void csr_spmv(const double a, const struct csr *A, const double *x,
              const double b, double *y) {
  for (unsigned i = 0; i < A->nrows; i++) {
    y[i] = b * y[i];
    for (unsigned j = A->offs[i]; j < A->offs[i + 1]; j++) {
      y[i] += a * A->vals[j] * x[A->cols[j] - A->base];
    }
  }
}
double l2norm(const double *x, const int n) {
  double norm = 0.0;
  for (int i = 0; i < n; i++) {
    norm += x[i] * x[i];
  }
  if (norm > 0)
    norm = sqrt(norm);
  return norm;
}
double glmax(const double *x, const int n) {
  double tmp = -1E10;
  for (int i = 0; i < n; i++) {
    tmp = fmax(tmp, x[i]);
  }
  return tmp;
}
double glmin(const double *x, const int n) {
  double tmp = 1E10;
  for (int i = 0; i < n; i++) {
    tmp = fmin(tmp, x[i]);
  }
  return tmp;
}
double glamax(const double *x, const int n) {
  double tmp = -1E10;
  for (int i = 0; i < n; i++) {
    tmp = fmax(tmp, fabs(x[i]));
  }
  return tmp;
}

void lsbench_matrix_print(const struct csr *A) {
  for (unsigned i = 0; i < A->nrows; i++) {
    for (unsigned s = A->offs[i], e = A->offs[i + 1]; s < e; s++)
      printf("%u %u %lf\n", i + A->base, A->cols[s], A->vals[s]);
  }
}

void lsbench_matrix_free(struct csr *A) {
  if (A) {
    tfree(A->offs);
    tfree(A->cols);
    tfree(A->vals);
  }
  tfree(A);
}

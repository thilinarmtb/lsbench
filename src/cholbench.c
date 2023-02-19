#include "cholbench-impl.h"
#include <assert.h>
#include <ctype.h>
#include <err.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void str_to_upper(char *up, const char *str) {
  size_t n = strnlen(str, BUFSIZ);
  for (unsigned i = 0; i < n; i++)
    up[i] = toupper(str[i]);
  up[n] = '\0';
}

static cholbench_solver_t str_to_solver(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "CUSOLVER") == 0) {
    return CHOLBENCH_SOLVER_CUSOLVER;
  } else if (strcmp(up, "HYPRE") == 0) {
    return CHOLBENCH_SOLVER_HYPRE;
  } else {
    warnx("Invalid solver: \"%s\". Defaulting to CUSOLVER.", str);
    return CHOLBENCH_SOLVER_CUSOLVER;
  }
}

static cholbench_ordering_t str_to_ordering(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "RCM") == 0) {
    return CHOLBENCH_ORDERING_RCM;
  } else if (strcmp(up, "AMD") == 0) {
    return CHOLBENCH_ORDERING_AMD;
  } else if (strcmp(up, "METIS") == 0) {
    return CHOLBENCH_ORDERING_METIS;
  } else {
    warnx("Invalid ordering: \"%s\". Defaulting to no ordering.", str);
    return CHOLBENCH_ORDERING_NONE;
  }
}

static cholbench_precision_t str_to_precision(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "FP64") == 0) {
    return CHOLBENCH_PRECISION_FP64;
  } else if (strcmp(up, "FP32") == 0) {
    return CHOLBENCH_PRECISION_FP32;
  } else if (strcmp(up, "FP16") == 0) {
    return CHOLBENCH_PRECISION_FP16;
  } else {
    warnx("Invalid precision: \"%s\". Defaulting to FP64.", str);
    return CHOLBENCH_PRECISION_FP64;
  }
}

static void print_help(int argc, char *argv[]) {
  printf("Usage: %s [OPTIONS]\n");
  printf("Options:\n");
  printf("  --matrix <FILE>\n");
  printf("  --solver <SOLVER>, Values: cusolver\n");
  printf("  --ordering <ORDERING>, Values: RCM, AMD, METIS\n");
  printf("  --precision <PRECISION>, Values: FP64, FP32, FP16\n");
  printf("  --verbose <VERBOSITY>, Values: 0, 1, 2, ...\n");
  printf("  --trials <TRIALS>, Values: 1, 2, ...\n");
  printf("  --help\n");
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

struct cholbench *cholbench_init(int argc, char *argv[]) {
  // Supported command line options.
  static struct option long_options[] = {
      {"matrix", required_argument, 0, 10},
      {"solver", required_argument, 0, 20},
      {"ordering", required_argument, 0, 30},
      {"precision", required_argument, 0, 40},
      {"verbose", required_argument, 0, 50},
      {"trials", required_argument, 0, 60},
      {"help", no_argument, 0, 70},
      {0, 0, 0, 0}};

  // Create the struct and set defauls.
  struct cholbench *cb = tcalloc(struct cholbench, 1);
  cb->matrix = NULL;
  cb->solver = CHOLBENCH_SOLVER_CUSOLVER;
  cb->ordering = CHOLBENCH_ORDERING_NONE;
  cb->precision = CHOLBENCH_PRECISION_FP64;
  cb->verbose = 0;
  cb->trials = 50;

  char bfr[BUFSIZ];
  // Parse the command line arguments.
  for (;;) {
    int c = getopt_long(argc, argv, "", long_options, NULL);
    if (c == -1)
      break;

    switch (c) {
    case 10:
      cb->matrix = strndup(optarg, BUFSIZ);
      break;
    case 20:
      strncpy(bfr, optarg, BUFSIZ);
      cb->solver = str_to_solver(bfr);
      break;
    case 30:
      strncpy(bfr, optarg, BUFSIZ);
      cb->ordering = str_to_ordering(bfr);
      break;
    case 40:
      strncpy(bfr, optarg, BUFSIZ);
      cb->precision = str_to_precision(bfr);
      break;
    case 50:
      cb->verbose = atoi(optarg);
      break;
    case 60:
      cb->trials = atoi(optarg);
      break;
    case 70:
      print_help(argc, argv);
      exit(EXIT_SUCCESS);
    default:
      print_help(argc, argv);
      exit(EXIT_FAILURE);
      break;
    }
  }

  // FIXME: Register these init functions.
  cusparse_init();
  hypre_init();

  return cb;
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
  case 1:
    hypre_bench(x, A, r, cb);
    break;
  default:
    errx(EXIT_FAILURE, "Unknown solver: %d.", cb->solver);
    break;
  }

  tfree(x), tfree(r);
}

void cholbench_finalize(struct cholbench *cb) {
  cusparse_finalize();
  hypre_finalize();
  if (cb)
    tfree(cb->matrix);
  tfree(cb);
}

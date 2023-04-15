#include "lsbench-impl.h"
#include <ctype.h>
#include <err.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>

#define MAX_BACKEND 32

static void str_to_upper(char *up, const char *str) {
  size_t n = strnlen(str, BUFSIZ);
  for (unsigned i = 0; i < n; i++)
    up[i] = toupper(str[i]);
  up[n] = '\0';
}

static lsbench_solver_t str_to_solver(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "CUSOLVER") == 0) {
    return LSBENCH_SOLVER_CUSOLVER;
  } else if (strcmp(up, "HYPRE") == 0) {
    return LSBENCH_SOLVER_HYPRE;
  } else if (strcmp(up, "AMGX") == 0) {
    return LSBENCH_SOLVER_AMGX;
  } else {
    warnx("Invalid solver: \"%s\". Defaulting to CUSOLVER.", str);
    return LSBENCH_SOLVER_CUSOLVER;
  }
}

static lsbench_ordering_t str_to_ordering(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "RCM") == 0) {
    return LSBENCH_ORDERING_RCM;
  } else if (strcmp(up, "AMD") == 0) {
    return LSBENCH_ORDERING_AMD;
  } else if (strcmp(up, "METIS") == 0) {
    return LSBENCH_ORDERING_METIS;
  } else {
    warnx("Invalid ordering: \"%s\". Defaulting to RCM.", str);
    return LSBENCH_ORDERING_RCM;
  }
}

static lsbench_precision_t str_to_precision(const char *str) {
  char up[BUFSIZ];
  str_to_upper(up, str);

  if (strcmp(up, "FP64") == 0) {
    return LSBENCH_PRECISION_FP64;
  } else if (strcmp(up, "FP32") == 0) {
    return LSBENCH_PRECISION_FP32;
  } else if (strcmp(up, "FP16") == 0) {
    return LSBENCH_PRECISION_FP16;
  } else {
    warnx("Invalid precision: \"%s\". Defaulting to FP64.", str);
    return LSBENCH_PRECISION_FP64;
  }
}

static void print_help(int argc, char *argv[]) {
  printf("Usage: %s [OPTIONS]\n");
  printf("Options:\n");
  printf("  --matrix <FILE>\n");
  printf("  --solver <SOLVER>, Values: hypre, amgx, cusolver\n");
  printf("  --ordering <ORDERING>, Values: RCM, AMD, METIS\n");
  printf("  --precision <PRECISION>, Values: FP64, FP32, FP16\n");
  printf("  --verbose <VERBOSITY>, Values: 0, 1, 2, ...\n");
  printf("  --trials <TRIALS>, Values: 1, 2, ...\n");
  printf("  --help\n");
}

static struct backend *backends[MAX_BACKEND];

struct lsbench *lsbench_init(int argc, char *argv[]) {
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
  struct lsbench *cb = tcalloc(struct lsbench, 1);
  cb->matrix = NULL;
  cb->solver = LSBENCH_SOLVER_NONE;
  cb->precision = LSBENCH_PRECISION_FP64;
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

  // Sanity checks
  if (cb->matrix == NULL || cb->solver == LSBENCH_SOLVER_NONE) {
    errx(EXIT_FAILURE,
         "Either input matrix or solver is not provided ! Try `--help`.");
  }

  // FIXME: Register these init functions.
#ifdef LSBENCH_CUSPARSE
  cusparse_init();
#endif
#ifdef LSBENCH_HYPRE
  hypre_init();
#endif
#ifdef LSBENCH_AMGX
  amgx_init();
#endif

  return cb;
}

void lsbench_bench(struct csr *A, const struct lsbench *cb) {
  unsigned m = A->nrows;
  double *x = tcalloc(double, m), *r = tcalloc(double, m);
  for (unsigned i = 0; i < m; i++)
    r[i] = i;

  // FIXME: Register these bench functions.
  switch (cb->solver) {
  case LSBENCH_SOLVER_CUSOLVER:
#ifdef LSBENCH_CUSPARSE
    cusparse_bench(x, A, r, cb);
#endif
    break;
  case LSBENCH_SOLVER_HYPRE:
#ifdef LSBENCH_HYPRE
    hypre_bench(x, A, r, cb);
#endif
    break;
  case LSBENCH_SOLVER_AMGX:
#ifdef LSBENCH_AMGX
    amgx_bench(x, A, r, cb);
#endif
    break;
  default:
    errx(EXIT_FAILURE, "Unknown solver: %d.", cb->solver);
    break;
  }

  tfree(x), tfree(r);
}

void lsbench_finalize(struct lsbench *cb) {
  // FIXME: Register these finalize functions.
#ifdef LSBENCH_CUSPARSE
  cusparse_finalize();
#endif
#ifdef LSBENCH_HYPRE
  hypre_finalize();
#endif
#ifdef LSBENCH_AMGX
  amgx_finalize();
#endif

  if (cb)
    tfree(cb->matrix);
  tfree(cb);
}

#undef MAX_BACKEND

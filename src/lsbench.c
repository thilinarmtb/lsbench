#include "lsbench-impl.h"
#include <ctype.h>
#include <err.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>

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
  } else if (strcmp(up, "CHOLMOD") == 0) {
    return LSBENCH_SOLVER_CHOLMOD;
  } else if (strcmp(up, "PARALMOND") == 0) {
    return LSBENCH_SOLVER_PARALMOND;
  } else {
    warnx("Invalid solver: \"%s\". Defaulting to HYPRE.", str);
    return LSBENCH_SOLVER_HYPRE;
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
  printf("  --solver <SOLVER>, Values: cusolver, hypre, amgx, cholmod\n");
  printf("  --ordering <ORDERING>, Values: RCM, AMD, METIS\n");
  printf("  --precision <PRECISION>, Values: FP64, FP32, FP16\n");
  printf("  --verbose <VERBOSITY>, Values: 0, 1, 2, ...\n");
  printf("  --trials <TRIALS>, Values: 1, 2, ...\n");
  printf("  --help\n");
}

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
  cb->trials = 100;

  // Parse the command line arguments.
  char bfr[BUFSIZ];
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

  // FIXME: Register these init functions so they can be called without the
  // ifdef.
#ifdef LSBENCH_CUSPARSE
  cusparse_init();
#endif
#ifdef LSBENCH_HYPRE
  hypre_init();
#endif
#ifdef LSBENCH_AMGX
  amgx_init();
#endif
#ifdef LSBENCH_CHOLMOD
  cholmod_init();
#endif
#ifdef LSBENCH_PARALMOND
  paralmond_init();
#endif

  return cb;
}

const char *lsbench_get_matrix_name(struct lsbench *cb) {
  return (const char *)cb->matrix;
}

void lsbench_bench(struct csr *A, const struct lsbench *cb) {
  unsigned m = A->nrows;
  double *x = tcalloc(double, 2 * m), *r = x + m;
  for (unsigned i = 0; i < m; i++)
    r[i] = i;

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
  case LSBENCH_SOLVER_CHOLMOD:
#ifdef LSBENCH_CHOLMOD
    cholmod_bench(x, A, r, cb);
#endif
    break;
  case LSBENCH_SOLVER_PARALMOND:
#ifdef LSBENCH_PARALMOND
    paralmond_bench(x, A, r, cb);
#endif
    break;
  default:
    errx(EXIT_FAILURE, "Unknown solver: %d.", cb->solver);
    break;
  }

  tfree(x);
}

void lsbench_finalize(struct lsbench *cb) {
#ifdef LSBENCH_CUSPARSE
  cusparse_finalize();
#endif
#ifdef LSBENCH_HYPRE
  hypre_finalize();
#endif
#ifdef LSBENCH_AMGX
  amgx_finalize();
#endif
#ifdef LSBENCH_CHOLMOD
  cholmod_finalize();
#endif
#ifdef LSBENCH_PARALMOND
  paralmond_finalize();
#endif

  if (cb)
    tfree(cb->matrix);
  tfree(cb);
}

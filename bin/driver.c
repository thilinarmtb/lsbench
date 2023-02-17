#include "cholbench.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  struct cholbench *cb = cholbench_init(argc, argv);

  struct csr *A = cholbench_matrix_read(cb);
  cholbench_bench(A, cb);
  cholbench_matrix_free(A);

  cholbench_finalize(cb);

  return 0;
}

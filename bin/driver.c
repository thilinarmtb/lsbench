#include "lsbench.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  struct lsbench *cb = lsbench_init(argc, argv);

  struct csr *A = lsbench_matrix_read(lsbench_get_matrix_name(cb));
  lsbench_bench(A, cb);
  lsbench_matrix_free(A);

  lsbench_finalize(cb);

  return 0;
}

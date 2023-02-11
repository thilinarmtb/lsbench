#include "cholbench.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  struct csr *A = cholbench_read(argv[1]);
  cholbench_print(A);
  cholbench_bench(A, 0, 1);
  cholbench_free(A);
  return 0;
}

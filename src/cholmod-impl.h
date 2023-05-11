#define csr_init SUFFIXED_NAME(csr_init)
static struct cholmod_csr *csr_init(struct csr *A, const struct lsbench *cb) {
  struct cholmod_csr *B = tcalloc(struct cholmod_csr, 1);

  uint nnz = A->offs[A->nrows];
  cholmod_triplet *T =
      allocate_triplet(A->nrows, A->nrows, nnz, -1, CHOLMOD_REAL, &cm);
  idx_t *Ti = (idx_t *)T->i, *Tj = (idx_t *)T->j;

  uint z = 0;
  double *Tx = (double *)T->x;
  for (uint i = 0; i < A->nrows; i++) {
    uint j;
    for (j = A->offs[i]; A->cols[j] - A->base < i; j++)
      ;
    for (uint je = A->offs[i + 1]; j < je; j++)
      Ti[z] = i, Tj[z] = A->cols[j] - A->base, Tx[z] = A->vals[j], z++;
  }
  T->nnz = z;

  // Convert triplet to CSC matrix.
  B->A = triplet_to_sparse(T, T->nnz, &cm);
  free_triplet(&T, &cm);

  // Factorize the matrix.
  B->L = analyze(B->A, &cm);
  factorize(B->A, B->L, &cm);

  B->r = zeros(A->nrows, 1, CHOLMOD_REAL, &cm);
  B->nr = A->nrows;

  return B;
}

#define csr_finalize SUFFIXED_NAME(csr_finalize)
static void csr_finalize(struct cholmod_csr *A) {
  if (A) {
    free_sparse(&A->A, &cm);
    free_factor(&A->L, &cm);
    free_dense(&A->r, &cm);
  }
  tfree(A);
}

#define bench SUFFIXED_NAME(bench)
static void bench(double *x, struct csr *A, const double *r,
                  const struct lsbench *cb) {
  struct cholmod_csr *B = csr_init(A, cb);
  double *rx = (double *)B->r->x;
  for (uint i = 0; i < B->nr; i++)
    rx[i] = r[i];

  // Warmup.
  for (unsigned i = 0; i < cb->trials; i++) {
    cholmod_dense *xd = solve(CHOLMOD_A, B->L, B->r, &cm);
    if (cb->verbose > 0) {
      double one[2] = {1, 0}, m1[2] = {-1, 0};
      cholmod_dense *rd = copy_dense(B->r, &cm);
      sdmult(B->A, 0, m1, one, xd, rd, &cm);
      printf("norm(b-Ax) = %e\n", norm_dense(rd, 0, &cm));
      free_dense(&rd, &cm);
    }
    free_dense(&xd, &cm);
  }

  // Time the solve.
  clock_t time = clock();
  for (unsigned i = 0; i < cb->trials; i++) {
    cholmod_dense *xd = solve(CHOLMOD_A, B->L, B->r, &cm);
    free_dense(&xd, &cm);
  }
  time = clock() - time;

  unsigned m = A->nrows, nnz = A->offs[m];
  printf("===matrix,n,nnz,trials,solver,ordering,elapsed===\n");
  printf("%s,%u,%u,%u,%u,%d,%.15lf\n", cb->matrix, m, nnz, cb->trials,
         cb->solver, cb->ordering, (double)time / CLOCKS_PER_SEC);
  gpu_stats(&cm);
  fflush(stdout);

  csr_finalize(B);
}
#undef bench
#undef csr_init
#undef csr_finalize

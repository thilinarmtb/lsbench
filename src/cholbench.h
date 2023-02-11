#ifndef _CHOLBENCH_
#define _CHOLBENCH_

#ifdef _cplusplus
extern "C" {
#endif

struct csr;

struct csr *cholbench_read(const char *fname);
void cholbench_bench(struct csr *A, unsigned solver, unsigned ntrials);
void cholbench_print(const struct csr *A);
void cholbench_free(struct csr *A);

#ifdef _cplusplus
}
#endif

#endif

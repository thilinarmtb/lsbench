#include "lsbench-impl.h"

#if defined(LSBENCH_GINKGO)
#include <ginkgo/ginkgo.hpp>

template <typename ValueType>
void print_vector(const std::string &name,
                  const gko::matrix::Dense<ValueType> *vec) {
  std::cout << name << " = [" << std::endl;
  for (int i = 0; i < vec->get_size()[0]; ++i) {
    std::cout << "    " << vec->at(i, 0) << std::endl;
  }
  std::cout << "];" << std::endl;
}

static std::shared_ptr<gko::matrix::Csr<double, int>>
csr_init(struct csr *A, const struct lsbench *cb) {
  auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

  unsigned m = A->nrows;
  unsigned nnz = A->offs[m];
  auto ginkgo_csr_host = gko::matrix::Csr<double, int>::create(
      exec->get_master(), gko::dim<2>{m, m}, nnz);
  // unsigned -> int since ginkgo also likes ints.
  for (unsigned i = 0; i < m + 1; i++)
    ginkgo_csr_host->get_row_ptrs()[i] = A->offs[i] + A->base;

  for (unsigned i = 0; i < nnz; i++)
    ginkgo_csr_host->get_col_idxs()[i] = A->cols[i];

  for (unsigned i = 0; i < nnz; i++)
    ginkgo_csr_host->get_values()[i] = A->vals[i];

  auto ginkgo_csr = gko::share(ginkgo_csr_host->clone(exec));

  return ginkgo_csr;
}

int ginkgo_bench(double *x, struct csr *A, const double *r,
                 const struct lsbench *cb) {

  unsigned m = A->nrows, nnz = A->offs[m];
  auto B = csr_init(A, cb);
  auto exec = B->get_executor();
  auto r_view = gko::array<double>::const_view(exec->get_master(), m, r);
  auto x_view = gko::array<double>::view(exec->get_master(), m, x);
  auto dense_x_host = gko::matrix::Dense<double>::create(
      exec, gko::dim<2>{m, 1}, std::move(x_view), 1);
  auto dense_r_host = gko::matrix::Dense<double>::create_const(
      exec, gko::dim<2>{m, 1}, std::move(r_view), 1);

  auto dense_r = dense_r_host->clone(exec);
  auto dense_x_init = dense_x_host->clone(exec);
  auto dense_x = dense_x_init->clone();
  auto solver =
      gko::solver::Bicgstab<double>::build()
          .with_preconditioner(
              gko::preconditioner::Jacobi<double>::build().on(exec))
          .with_criteria(gko::stop::ImplicitResidualNorm<double>::build()
                             .with_baseline(gko::stop::mode::initial_resnorm)
                             .with_reduction_factor(1e-4)
                             .on(exec))
          .on(exec)
          ->generate(B);
  // Warmup
  for (unsigned i = 0; i < cb->trials; i++) {
    dense_x->copy_from(dense_x_init);
    solver->apply(dense_r, dense_x);
  }

  // This adds a simple logger that only reports convergence state at the end
  // of the solver. Specifically it captures the last residual norm, the
  // final number of iterations, and the converged or not converged status.

  // Some shortcuts
  using ValueType = double;
  using RealValueType = gko::remove_complex<ValueType>;
  using IndexType = int;

  using vec = gko::matrix::Dense<ValueType>;
  using real_vec = gko::matrix::Dense<RealValueType>;
  using mtx = gko::matrix::Csr<ValueType, IndexType>;
  using cg = gko::solver::Cg<ValueType>;

  std::shared_ptr<gko::log::Convergence<ValueType>> convergence_logger =
      gko::log::Convergence<ValueType>::create();
  solver->add_logger(convergence_logger);

  // Time the solve
  double time = 0;
  for (unsigned i = 0; i < cb->trials; i++) {
    dense_x->copy_from(dense_x_init);
    exec->synchronize();
    clock_t t = clock();
    solver->apply(dense_r, dense_x);
    exec->synchronize();
    t = clock() - t;
    time += static_cast<double>(t);
  }

  dense_x_host->copy_from(dense_x);

  std::cout << "Residual norm sqrt(r^T r):\n";
  write(std::cout, gko::as<vec>(convergence_logger->get_residual_norm()));
  std::cout << "Number of iterations "
            << convergence_logger->get_num_iterations() << std::endl;
  std::cout << "Convergence status " << std::boolalpha
            << convergence_logger->has_converged() << std::endl;

  printf("===matrix,n,nnz,trials,solver,ordering,elapsed===\n");
  printf("%s,%u,%u,%u,%u,%d,%.15lf\n", cb->matrix, m, nnz, cb->trials,
         cb->solver, cb->ordering, time / CLOCKS_PER_SEC);
  fflush(stdout);
  return 0;
}

#else
int ginkgo_bench(double *x, struct csr *A, const double *r,
                 const struct lsbench *cb) {
  return 1;
}
#endif

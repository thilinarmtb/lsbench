## Introduction

`lsbench` is a repository containing benchmark codes for various high performant
linear solvers on CPUs and GPUs. It can read a sparse matrix and solve it using
`cuSparse`, `AmgX`, `Hypre` or `CHOLMOD`. An example on how to use the API is in
`bin/driver.c`. A few test matrices are available in `tests/` directory. Currently,
all the benchmarks are performend in double precision.

## Building lsbench

`lsbench` by defaults download and builds `CHOLMOD`. You can enable/disable
solvers (`AmgX`, `cuSparse`, etc.) when configuring the `cmake` build using
`ENABLE_<SOLVER>=ON|OFF`.

### Build requirements

- cmake (>= 3.18)
- CUDAToolkit (>= 11.0) if using AmgX, cuSparse or Hypre with Cuda backend)

### Build instructions

1. Clone the repo first using `git` and `cd` into the repository:
```sh
git clone https://github.com/thilinarmtb/lsbench.git
cd lsbench
```

2. Then you can use `cmake` to build the benchmarks:
```sh
mkdir build && cd build 
cmake .. -DCMAKE_C_COMPILER=<c-compiler> -DCMAKE_CXX_COMPILER=<cxx-compiler> \
  -DCMAKE_INSTALL_PREFIX=<lsbench-install-dir> \
  -DENABLE_CUSPARSE=ON -DENABLE_AMGX=OFF
make -j8 install
cd -
```

You can add the `bin` directory to the `PATH` variable to access the binary
without the full path:
```sh
export PATH=${PATH}/<lsbench-install-dir>/bin
```

## Running the benchmarks

Once the benchmarks are built, they can be run using the following command:
Do `driver --help` to see available options (not everything is implemented
right now).
```sh
driver --solver cholmod --test ./tests/I1_05x05.txt --verbose=1
```

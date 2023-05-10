module load cmake
module load PrgEnv-gnu
module load amd-mixed

module load openblas

module load craype-accel-amd-gfx90a
module unload cray-libsci

export MPICH_GPU_SUPPORT_ENABLED=1

module list

#Currently Loaded Modules:
#  1) craype-x86-trento                       7) gcc/12.2.0         13) darshan-runtime/3.4.0
#  2) libfabric/1.15.2.0                      8) cmake/3.23.2       14) hsi/default
#  3) craype-network-ofi                      9) craype/2.7.19      15) DefApps/default
#  4) perftools-base/22.12.0                 10) cray-dsmml/0.2.2   16) amd-mixed/5.3.0
#  5) xpmem/2.5.2-2.4_3.30__gd0f7936.shasta  11) cray-mpich/8.1.23  17) openblas/0.3.17
#  6) cray-pmi/6.1.8                         12) PrgEnv-gnu/8.3.3   18) craype-accel-amd-gfx90a


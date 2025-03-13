# Deformation Transfer algorithm (Homework)

Efficient implementation with Eigen, OpenMesh and multi-threading. Needs C++23.

## Build
```
mkdir build && cd build
cmake ..
make
```

## Usage
```
./deformtrans <s0.obj> <s1.obj> <t0.obj> <t1.obj>
```

## Acknowledgement
Thanks to [this repository](https://github.com/mickare/Deformation-Transfer-for-Triangle-Meshes) for key mathematical derivation.

## Reference
[Deformation Transfer for Triangle Meshes](https://people.csail.mit.edu/sumner/research/deftransfer/Sumner2004DTF.pdf)

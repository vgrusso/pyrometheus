# Notes: Building Mutation++ Python Bindings on HPC

## Install nanobind

### Clone repository

```bash
git clone --recursive https://github.com/wjakob/nanobind.git
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

### Configure and build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Install locally

```bash
cmake --install build --prefix $HOME/.local
```

This installs nanobind into:

```text
$HOME/.local/nanobind
```

including the CMake package files:

```text
$HOME/.local/nanobind/cmake/nanobind-config.cmake
$HOME/.local/nanobind/cmake/nanobind-config-version.cmake
```

---

## Configure Mutation++

Load this CMake on ISAAC 

```bash
module load cmake/3.30.5-gcc
```

Set path installation prefix:

```bash
export CMAKE_PREFIX_PATH=$HOME/.local:$CMAKE_PREFIX_PATH
```

Configure build adding nanobind (do not forget!):

```bash
cmake -S . -B build \
  -DSKBUILD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -Dnanobind_DIR=$HOME/.local/nanobind/cmake \
  -DPython_EXECUTABLE=$(which python)
```

## Build Mutation++

Build normally:

```bash
cmake --build build -j
```

After a successful build, Python extension module is at:

```text
build/interface/python/_mutationpp.cpython-39-x86_64-linux-gnu.so
```
---

## Issue I faced

The Python package located in:

```text
interface/python/mutationpp/
```

contains:

```python
from ._mutationpp import *
```

Therefore Python expects `_mutationpp` to reside inside the package directory:

```text
interface/python/mutationpp/
```

However, the current build places the shared library in:

```text
build/interface/python/
```

As a result:

```bash
import mutationpp
```

fails with:

```bash
ModuleNotFoundError: No module named 'mutationpp._mutationpp'
```

even though the extension module was successfully compiled.

---

## Current Workaround

Copy the generated shared library into the package directory:

```bash
cp build/interface/python/_mutationpp.*.so \
   interface/python/mutationpp/
```

This places `_mutationpp` where nanobind is gonna look for it.

---

## Complete Build Procedure I used (with a lil workaround, God forgive me)

```bash
module load cmake/3.30.5-gcc && \

rm -rf build && \

export CMAKE_PREFIX_PATH=$HOME/.local:$CMAKE_PREFIX_PATH && \

cmake -S . -B build \
  -DSKBUILD=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -Dnanobind_DIR=$HOME/.local/nanobind/cmake \
  -DPython_EXECUTABLE=$(which python) && \

cmake --build build -j && \

cp build/interface/python/_mutationpp.*.so \
   interface/python/mutationpp/
```

---

## Checks

Ensure the package path is visible:

```bash
export PYTHONPATH=/path/to/Mutationpp/interface/python:$PYTHONPATH
```

Test the import:

```bash
python -c "import mutationpp; print(mutationpp)"
```

Expected result:

```text
<module 'mutationpp' ...>
```

with no import errors.

---

## Potential Improvement

The manual copy is ugly and need to be fixed

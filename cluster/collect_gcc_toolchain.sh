#!/usr/bin/env bash

set -e

OUTPUT_DIR="gcc-toolchain"
mkdir -p $OUTPUT_DIR/{bin,lib/gcc,libexec/gcc,include,lib64}

echo "Collecting minimal GCC toolchain..."

cp -v /usr/bin/gcc $OUTPUT_DIR/bin/
cp -v /usr/bin/g++ $OUTPUT_DIR/bin/

echo "Copying GCC library files..."
cp -rv /usr/lib/gcc/* $OUTPUT_DIR/lib/gcc/

echo "Copying GCC executables (cc1plus, etc)..."
echo "Note: cc1plus is already in /usr/lib/gcc structure, no separate copy needed"

echo "Copying system include files..."
cp -rv /usr/include/* $OUTPUT_DIR/include/

echo "Copying essential runtime libraries from /usr/lib/x86_64-linux-gnu..."
mkdir -p $OUTPUT_DIR/lib64

cp -v /usr/lib/x86_64-linux-gnu/crt*.o $OUTPUT_DIR/lib64/ 2>/dev/null || true
cp -v /usr/lib/x86_64-linux-gnu/libstdc++.so* $OUTPUT_DIR/lib64/
cp -v /usr/lib/x86_64-linux-gnu/libgcc_s.so* $OUTPUT_DIR/lib64/
cp -v /usr/lib/x86_64-linux-gnu/libc.so* $OUTPUT_DIR/lib64/ 2>/dev/null || true
cp -v /usr/lib/x86_64-linux-gnu/libm.so* $OUTPUT_DIR/lib64/ 2>/dev/null || true

echo "Creating tarball..."
tar -czf gcc-toolchain.tar.gz $OUTPUT_DIR

echo "Done! Created gcc-toolchain.tar.gz"
echo "Size: $(du -h gcc-toolchain.tar.gz | cut -f1)"
echo ""
echo "Upload to cluster with:"
echo "scp gcc-toolchain.tar.gz zchai33@login-ice.pace.gatech.edu:~/scratch/"
echo ""
echo "Extract on cluster with:"
echo "cd ~/scratch && tar -xzf gcc-toolchain.tar.gz"

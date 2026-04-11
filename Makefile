# Compilers
CXX := g++-11
NVCC := /usr/local/cuda-11.7/bin/nvcc

# Flags
CXXFLAGS := -std=c++17 
NVCCFLAGS := -arch=sm_75 -ccbin /usr/bin/g++-11

# Targets
all: bhut bhut_cuda

# Serial Implementation
bhut: bhut.cpp
	$(CXX) $(CXXFLAGS) bhut.cpp -o bhut

# CUDA Implementation
bhut_cuda: bhut.cu kernels.o kernels.cuh
	$(NVCC) $(NVCCFLAGS) bhut.cu kernels.o -o bhut_cuda

# Shared object file for kernels
kernels.o: kernels.cu kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c kernels.cu -o kernels.o

# Test: body_reduce_kernel
bhut_cpu.o: bhut_cpu.cpp bhut_cpu.h
	$(CXX) $(CXXFLAGS) -c bhut_cpu.cpp -o bhut_cpu.o


test: kernels.o bhut_cpu.o test/test_tree.cu kernels.cuh bhut_cpu.h
	$(NVCC) $(NVCCFLAGS) -I. -c test/test_tree.cu -o test/test_tree.o
	$(NVCC) $(NVCCFLAGS) kernels.o bhut_cpu.o test/test_tree.o -o test/test_tree
	rm -f test/test_tree.o


# Cleanup
clean:
	rm -f bhut bhut_cuda bhut_cpu.o kernels.o test/*.o test/test_reduce test/test_tree
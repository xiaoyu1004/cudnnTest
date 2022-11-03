#include "common.h"

static constexpr int kNbThreadsPerBlockReduce = 512;

template <typename T>
__device__ void ReductionMax(int tid, T *sdata, int len)
{
    auto pow2 = len;
    if (pow2 & (pow2 - 1))
    {
        while (pow2 & (pow2 - 1))
        {
            pow2 &= (pow2 - 1);
        }
        if (tid >= pow2)
        {
            sdata[tid - pow2] = max(sdata[tid - pow2], sdata[tid]);
        }
        __syncthreads();
    }

    if (pow2 == 2048)
    {
        if (tid < 1024)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + 1024]);
        }
        __syncthreads();
    }

    if (pow2 >= 1024)
    {
        if (tid < 512)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + 512]);
        }
        __syncthreads();
    }

    if (pow2 >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + 256]);
        }
        __syncthreads();
    }

    if (pow2 >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + 128]);
        }
        __syncthreads();
    }

    if (pow2 >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + 64]);
        }
        __syncthreads();
    }

    if (tid < 32)
    {
        volatile T *vsdata = sdata;
        if (pow2 >= 64 && tid < 32)
        {
            vsdata[tid] = max(vsdata[tid], vsdata[tid + 32]);
        }

        if (pow2 >= 32 && tid < 16)
        {
            vsdata[tid] = max(vsdata[tid], vsdata[tid + 16]);
        }

        if (pow2 >= 16 && tid < 8)
        {
            vsdata[tid] = max(vsdata[tid], vsdata[tid + 8]);
        }

        if (pow2 >= 8 && tid < 4)
        {
            vsdata[tid] = max(vsdata[tid], vsdata[tid + 4]);
        }

        if (pow2 >= 4 && tid < 2)
        {
            vsdata[tid] = max(vsdata[tid], vsdata[tid + 2]);
        }

        if (pow2 >= 2 && tid < 1)
        {
            vsdata[tid] = max(vsdata[tid], vsdata[tid + 1]);
        }
    }
}

template <typename T>
void __global__ reduceMax4DW(int dim_n,
                             int dim_c,
                             int dim_h,
                             int dim_w,
                             const T *A,
                             T *C)
{
    int tid = threadIdx.x;
    int base_idx = blockIdx.x * dim_w;

    __shared__ T sdata[kNbThreadsPerBlockReduce];

    T max = 0;
    if (tid < dim_w)
    {
        max = A[base_idx + tid];
    }
    sdata[tid] = max;
    __syncthreads();
    ReductionMax<T>(tid, sdata, kNbThreadsPerBlockReduce);
    __syncthreads();
    max = sdata[0];

    for (int offset = kNbThreadsPerBlockReduce + tid; offset < dim_w; offset += kNbThreadsPerBlockReduce)
    {
        sdata[tid] = A[base_idx + offset];
        __syncthreads();
        ReductionMax<T>(tid, sdata, kNbThreadsPerBlockReduce);
        __syncthreads();
        T max_now = sdata[0];
        if (max_now > max)
        {
            max = max_now;
        }
    }

    int c_idx = blockIdx.x;
    if (tid == 0)
    {
        C[c_idx] = max;
    }
}

template <typename T>
void __global__ reduceMax4DC(int dim_n,
                              int dim_c,
                              int dim_h,
                              int dim_w,
                              const T *A,
                              T *C)
{
    int tid = threadIdx.x;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int base_idx = by * dim_c * dim_h + bx;

    __shared__ T sdata[kNbThreadsPerBlockReduce];

    T max = 0;
    if (tid < dim_c)
    {
        max = A[base_idx + tid * dim_h];
    }
    sdata[tid] = max;
    __syncthreads();
    ReductionMax<T>(tid, sdata, kNbThreadsPerBlockReduce);
    __syncthreads();
    max = sdata[0];

    for (int offset = kNbThreadsPerBlockReduce + tid; offset < dim_c; offset += kNbThreadsPerBlockReduce)
    {
        sdata[tid] = A[base_idx + offset * dim_h];
        __syncthreads();
        ReductionMax<T>(tid, sdata, kNbThreadsPerBlockReduce);
        __syncthreads();
        T max_now = sdata[0];
        if (max_now > max)
        {
            max = max_now;
        }
    }

    int c_idx = by * dim_h + bx;
    if (tid == 0)
    {
        C[c_idx] = max;
    }
}

template <typename T>
void reduceMax4D(int dim_n,
                 int dim_c,
                 int dim_h,
                 int dim_w,
                 T *workspace,
                 const T *A,
                 T *C)
{
    {
        dim3 dimBlock(kNbThreadsPerBlockReduce, 1, 1);
        dim3 dimGrid(dim_n * dim_c * dim_h, 1, 1);
        reduceMax4DW<T><<<dimGrid, dimBlock>>>(dim_n, dim_c, dim_h, dim_w, A, workspace);
    }
    {
        dim3 dimBlock(kNbThreadsPerBlockReduce, 1, 1);
        dim3 dimGrid(dim_h, dim_n, 1);
        reduceMax4DC<T><<<dimGrid, dimBlock>>>(dim_n, dim_c, dim_h, 1, workspace, C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_POST_KERNEL_CHECK;
}

template void reduceMax4D<float>(int dim_n,
                                 int dim_c,
                                 int dim_h,
                                 int dim_w,
                                 float *workspace,
                                 const float *A,
                                 float *C);
#include "common.h"

static constexpr unsigned int kNbThreadsPerBlockReduce = 512;

template <typename T>
__device__ void ReductionMax(unsigned int tid, T *sdata, unsigned int len)
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
void __global__ reduceMax5DW(unsigned int dim_n,
                             unsigned int dim_c,
                             unsigned int dim_d,
                             unsigned int dim_h,
                             unsigned int dim_w,
                             const T *A,
                             T *C)
{
    unsigned int tid = threadIdx.x;

    unsigned int bx = blockIdx.x;
    unsigned int base_idx = bx * dim_w;

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

    unsigned int loopCnt = (dim_w + kNbThreadsPerBlockReduce - 1) / kNbThreadsPerBlockReduce;
    for (unsigned int i = 0; i < loopCnt; ++i)
    {
        unsigned int offset = i * kNbThreadsPerBlockReduce + tid;
        float val = 0.f;
        if (offset < dim_w)
        {
            val = A[base_idx + offset];
        }
        sdata[tid] = val;
        __syncthreads();
        ReductionMax<T>(tid, sdata, kNbThreadsPerBlockReduce);
        __syncthreads();
        T max_now = sdata[0];
        if (max_now > max)
        {
            max = max_now;
        }
    }

    unsigned int c_idx = bx;
    if (tid == 0)
    {
        C[c_idx] = max;
    }
}

#define Ceil(v, b) ((((v) + (b)-1) / (b)))

template <typename T>
void __global__ reduceMax5DD(unsigned int dim_n,
                             unsigned int dim_c,
                             unsigned int dim_d,
                             unsigned int dim_h,
                             unsigned int dim_w,
                             const T *A,
                             T *C)
{
    unsigned int tid = threadIdx.x;

    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int bz = blockIdx.z;

    unsigned int base_idx = bz * dim_c * dim_d * dim_h + by * dim_d * dim_h + bx;

    __shared__ T sdata[kNbThreadsPerBlockReduce];

    T max = 0;
    if (tid < dim_d)
    {
        max = A[base_idx + tid * dim_h];
    }
    sdata[tid] = max;
    __syncthreads();
    ReductionMax<T>(tid, sdata, kNbThreadsPerBlockReduce);
    __syncthreads();
    max = sdata[0];

    unsigned int loopCnt = (dim_d + kNbThreadsPerBlockReduce - 1) / kNbThreadsPerBlockReduce;
    for (unsigned int i = 0; i < loopCnt; ++i)
    {
        unsigned int offset = i * kNbThreadsPerBlockReduce + tid;
        float val = 0.f;
        if (offset < dim_d)
        {
            val = A[base_idx + offset * dim_h];
        }
        sdata[tid] = val;
        __syncthreads();
        ReductionMax<T>(tid, sdata, kNbThreadsPerBlockReduce);
        __syncthreads();
        T max_now = sdata[0];
        if (max_now > max)
        {
            max = max_now;
        }
    }

    unsigned int c_idx = bz * dim_c * dim_h + by * dim_h + bx;
    if (tid == 0)
    {
        C[c_idx] = max;
    }
}

template <typename T>
void reduceMax5D(unsigned int dim_n,
                 unsigned int dim_c,
                 unsigned int dim_d,
                 unsigned int dim_h,
                 unsigned int dim_w,
                 T *workspace,
                 const T *A,
                 T *C)
{
    {
        dim3 dimBlock(kNbThreadsPerBlockReduce, 1, 1);
        dim3 dimGrid(dim_n * dim_c * dim_d * dim_h, 1, 1);
        reduceMax5DW<T><<<dimGrid, dimBlock>>>(dim_n, dim_c, dim_d, dim_h, dim_w, A, workspace);
        // reduceMax5DW<T><<<dimGrid, dimBlock>>>(dim_n, dim_c, dim_d, dim_h, dim_w, A, C);
    }
    {
        dim3 dimBlock(kNbThreadsPerBlockReduce, 1, 1);
        dim3 dimGrid(dim_h, dim_c, dim_n);
        reduceMax5DD<T><<<dimGrid, dimBlock>>>(dim_n, dim_c, dim_d, dim_h, 1, workspace, C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_POST_KERNEL_CHECK;
}

template void reduceMax5D<float>(unsigned int dim_n,
                                 unsigned int dim_c,
                                 unsigned int dim_d,
                                 unsigned int dim_h,
                                 unsigned int dim_w,
                                 float *workspace,
                                 const float *A,
                                 float *C);
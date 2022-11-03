#include "reduce/reduce.h"
#include "common.h"

#define ENABLE_PRINT 0

#if ENABLE_PRINT
#define PRINT_TENSOR(device, ptr, flag)                                                                                                                      \
    std::cout << #device << ":" << std::endl;                                                                                                                \
    for (int n = 0; n < input_n##flag; ++n)                                                                                                                  \
    {                                                                                                                                                        \
        for (int i = 0; i < input_h##flag; ++i)                                                                                                              \
        {                                                                                                                                                    \
            for (int j = 0; j < input_c##flag; ++j)                                                                                                          \
            {                                                                                                                                                \
                for (int k = 0; k < input_w##flag; ++k)                                                                                                      \
                {                                                                                                                                            \
                    std::cout << ptr[n * input_h##flag * input_w##flag * input_c##flag + j * input_h##flag * input_w##flag + i * input_w##flag + k] << "\t"; \
                }                                                                                                                                            \
                std::cout << "\t\t";                                                                                                                         \
            }                                                                                                                                                \
            std::cout << std::endl;                                                                                                                          \
        }                                                                                                                                                    \
        std::cout << "\n";                                                                                                                                   \
    }
#else
#define PRINT_TENSOR(device, ptr, flag)
#endif

void ReduceMax4DTest()
{
    int input_n0 = 1;
    int input_c0 = 4;
    int input_h0 = 4;
    int input_w0 = 4;

    int input_n1 = 1;
    int input_c1 = 1;
    int input_h1 = 4;
    int input_w1 = 1;

    int aSize = input_n0 * input_c0 * input_h0 * input_w0;
    float *h_a = new float[aSize]{0.f};
    for (int i = 0; i < aSize; ++i)
    {
        h_a[i] = i;
    }

    int cSize = input_n1 * input_c1 * input_h1 * input_w1;
    float *h_c = new float[cSize]{0.f};
    float *h_dnn_c = new float[cSize]{0.f};

    size_t h_indices = 0;

    PRINT_TENSOR(cpu, h_a, 0);

    float *d_a;
    CUDA_CHECK(cudaMalloc(&d_a, aSize * sizeof(float)));
    float *d_c;
    CUDA_CHECK(cudaMalloc(&d_c, cSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, aSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, cSize * sizeof(float), cudaMemcpyHostToDevice));

    // cuda
    int work_space_bytes = input_n0 * input_c0 * input_h0 * sizeof(float);
    float *work_space;
    CUDA_CHECK(cudaMalloc(&work_space, work_space_bytes));
    reduceMax4D<float>(input_n0, input_c0, input_h0, input_w0, work_space, d_a, d_c);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, cSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_c, 0, cSize * sizeof(float)));
    PRINT_TENSOR(gpu, h_c, 1);

    // cudnn
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));
    cudnnReduceTensorDescriptor_t reduceTensorDesc;
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduceTensorDesc));
    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(reduceTensorDesc, CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES));

    cudnnTensorDescriptor_t aDesc;
    cudnnTensorDescriptor_t cDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&aDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n0, input_c0, input_h0, input_w0));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n1, input_c1, input_h1, input_w1));

    size_t *d_indices;
    size_t indicesSizeInBytes;
    CUDNN_CHECK(cudnnGetReductionIndicesSize(handle, reduceTensorDesc, aDesc, cDesc, &indicesSizeInBytes));
    CUDA_CHECK(cudaMalloc(&d_indices, indicesSizeInBytes));

    float *workspace;
    size_t workspaceSizeInBytes;
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(handle, reduceTensorDesc, aDesc, cDesc, &workspaceSizeInBytes));
    CUDA_CHECK(cudaMalloc(&workspace, workspaceSizeInBytes));

    float alpha = 1.f;
    float beta = 0.f;

    CUDNN_CHECK(cudnnReduceTensor(handle, reduceTensorDesc, d_indices, indicesSizeInBytes, workspace, workspaceSizeInBytes, &alpha, aDesc, d_a, &beta, cDesc, d_c));
    CUDA_CHECK(cudaMemcpy(h_dnn_c, d_c, cSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_indices, d_indices, sizeof(float), cudaMemcpyDeviceToHost));

    PRINT_TENSOR(cudnn, h_dnn_c, 1);
    std::cout << "h_indices: " << h_indices << std::endl;

    for (int i = 0; i < cSize; ++i)
    {
        float err = std::abs(h_dnn_c[i] - h_c[i]);
        if (err > 1e-3)
        {
            std::cout << "ERROR! h_dnn_c[" << i << "]=" << h_dnn_c[i] << " vs h_c[" << i << "]=" << h_c[i] << std::endl;
            std::terminate();
        }
    }
    std::cout << "compare pass!" << std::endl;

    // free
    delete[] h_a;
    delete[] h_c;
    delete[] h_dnn_c;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(work_space));

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reduceTensorDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(aDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(cDesc));
}

int main()
{
    ReduceMax4DTest();
    return 0;
}
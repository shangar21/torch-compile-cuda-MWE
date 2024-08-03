#include <torch/extension.h>

__global__ void matmul_kernel(float* a, float* b, float* c, int D) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < D && col < D) {
        for (int k = 0; k < D; ++k) {
            c[row * D + col] += a[row * D + k] * b[k * D + col];
        }
    }
}

torch::Tensor matDxD(torch::Tensor a, torch::Tensor b) {
    const auto D = a.size(0);
    auto c = torch::zeros({D, D}, a.options());

    const int threads = 16;
    const dim3 blocks((D + threads - 1) / threads, (D + threads - 1) / threads);
    const dim3 threadsPerBlock(threads, threads);

    matmul_kernel<<<blocks, threadsPerBlock>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        D
    );

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matDxD", &matDxD, "Matrix multiplication on GPU");
}


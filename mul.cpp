#include <torch/extension.h>

torch::Tensor matDxD(
		torch::Tensor a,
		torch::Tensor b
){
	return torch::matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
	m.def("matDxD", &matDxD, "matDxD");
}


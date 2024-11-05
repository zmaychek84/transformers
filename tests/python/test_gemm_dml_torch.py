import torch
import ryzenai_torch_cpp as rai
import RyzenAI
import qlinear


def print_dlls():  # from https://stackoverflow.com/questions/5553917/how-to-list-all-dlls-loaded-by-a-process-with-python
    import psutil, os

    p = psutil.Process(os.getpid())
    for dll in p.memory_maps():
        print(dll.path)


def test_gemm_dml_torch():
    M, K, N = 1, 64, 128
    grp_size = 32
    target_err_percent = 1.0

    a_gpu = torch.ones(1, M, K).to(torch.float) * 13

    a_cpu = a_gpu.to(torch.bfloat16)
    b_cpu = torch.ones(N, K).to(torch.float)
    bias = torch.zeros(M, N).to(torch.float)

    gemm_cpu = qlinear.QLinearPerGrp(
        in_features=M,
        out_features=N,
        bias=False,
        device="cpu",
        w_bit=4,
        group_size=grp_size,
    )

    gemm_cpu.weight = b_cpu
    gemm_cpu.bias = bias
    gemm_cpu.quantize_weights()  ## Generating Sclaes

    b_gpu = gemm_cpu.qweight
    scales_gpu = gemm_cpu.scales.reshape(N, K // grp_size).to(torch.float)
    zeros_gpu = gemm_cpu.qzeros.reshape(N, K // grp_size).to(torch.uint8)

    gemm_cpu.initialize_parameters()
    out_cpu = gemm_cpu(a_cpu)

    dml_gemm = rai.dml_gemm_torch(False, M, K, N, "Float", "Uint4", "Float")
    dml_gemm.initialize_params(b_gpu, zeros_gpu, scales_gpu, bias, grp_size)
    out_gpu = dml_gemm.execute(a_gpu)

    out_gpu = out_gpu.reshape(out_cpu.shape).to(out_cpu.dtype)

    result = torch.allclose(out_cpu, out_gpu, target_err_percent / 100)

    assert result == True


if __name__ == "__main__":
    test_gemm_dml_torch()

import os
import time
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from artchitecture import HydroTransformerNet

BASE_DIR = "."
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def export_to_onnx():
    print("[*] Initializing Day 6 Task 1-6: ONNX Export & Verification...")
    device = torch.device('cpu')
    
    # Load model architecture and weights
    model = HydroTransformerNet(hidden_dim=64, num_layers=3, num_heads=4).to(device)
    
    checkpoint_path = os.path.join(BASE_DIR, "models", "best_hydro_vit_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[*] Successfully loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint not found at {checkpoint_path}. Exporting with initialized weights.")
    model.eval()

    # Create dummy input tensors for tracing
    dummy_acoustic = torch.randn(1, 1, 10, 10, device=device)
    dummy_tabular = torch.randn(1, 6, device=device)
    
    onnx_path = os.path.join(ARTIFACTS_DIR, "hydro_vit.onnx")
    
    # Export with dynamic axes for batch size scalability
    torch.onnx.export(
        model,
        (dummy_acoustic, dummy_tabular),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['acoustic_input', 'tabular_input'],
        output_names=['output'],
        dynamic_axes={
            'acoustic_input': {0: 'batch_size'},
            'tabular_input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Load exported model and clear stale intermediate value_info
    onnx_model = onnx.load(onnx_path)
    del onnx_model.graph.value_info[:]
    onnx.save(onnx_model, onnx_path)

    # Verify ONNX model structure
    onnx.checker.check_model(onnx_model)
    print(f"[*] ONNX model successfully exported and verified at {onnx_path}")
    return onnx_path


def benchmark_onnx_runtime(onnx_path):
    print("[*] Initializing Tasks 7-9: Latency Benchmarking (PyTorch vs ONNX Runtime)...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    acoustic_np = np.random.randn(1, 1, 10, 10).astype(np.float32)
    tabular_np = np.random.randn(1, 6).astype(np.float32)
    
    # Warmup
    for _ in range(50):
        session.run(None, {'acoustic_input': acoustic_np, 'tabular_input': tabular_np})
        
    # Benchmark ONNX Runtime over 1,000 iterations
    start_time = time.time()
    for _ in range(1000):
        session.run(None, {'acoustic_input': acoustic_np, 'tabular_input': tabular_np})
    onnx_duration = time.time() - start_time
    p99_onnx = (onnx_duration / 1000.0) * 1000  # ms
    print(f"[*] ONNX Runtime Average Latency over 1,000 runs: {p99_onnx:.4f} ms")


def perform_int8_quantization(model_fp32_path):
    print("[*] Initializing Tasks 10-14: INT8 Dynamic Quantization...")
    model_quant_path = os.path.join(ARTIFACTS_DIR, "hydro_vit_int8.onnx")
    
    # Bypass strict shape inference to prevent multi-modal tensor dimension conflicts
    quantize_dynamic(
        model_input=model_fp32_path,
        model_output=model_quant_path,
        weight_type=QuantType.QInt8,
        extra_options={"DisableShapeInference": True}
    )
    
    fp32_size = os.path.getsize(model_fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(model_quant_path) / (1024 * 1024)
    print(f"[*] FP32 Model Size: {fp32_size:.2f} MB | INT8 Quantized Model Size: {int8_size:.2f} MB")
    print(f"[*] INT8 Quantization completed successfully. Saved to {model_quant_path}")


if __name__ == '__main__':
    fp32_path = export_to_onnx()
    benchmark_onnx_runtime(fp32_path)
    perform_int8_quantization(fp32_path)
"""
NEXLYRA MLOPS: ONNX EXPORT, INT8 QUANTIZATION & LATENCY BENCHMARK
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_hydro_model.pth")
ONNX_OUTPUT_PATH = os.path.join(BASE_DIR,"artifacts", "hydro_cavnet.onnx")
QUANT_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "hydro_cavnet_int8.pth")

'''
print(f"Model path{MODEL_PATH}")
print(f"Model path{ONNX_OUTPUT_PATH}")
print(f"Model path{QUANT_MODEL_PATH}")

'''

# Canonical HydroCavitationNet Multi-Modal Architecture
class HydroCavitationNet(nn.Module):
    def __init__(self, tabular_dim=6):
        super(HydroCavitationNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.GELU()

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(p=0.35)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(1600 + 64, 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.GELU()
        )
        self.classifier = nn.Linear(64, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, spec_x, tabular_x):
        x_spatial = self.act1(self.bn1(self.conv1(spec_x)))
        x_spatial = self.act2(self.bn2(self.conv2(x_spatial)))
        x_spatial = self.pool(x_spatial)
        x_spatial = self.flatten(x_spatial)

        x_tab = self.tabular_mlp(tabular_x)
        fused = torch.cat((x_spatial, x_tab), dim=1)
        fused_features = self.fusion_mlp(fused)
        return self.classifier(fused_features)

# Load Model
device = torch.device("cpu")
model = HydroCavitationNet(tabular_dim=6).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[SUCCESS] Loaded model checkpoint from '{MODEL_PATH}'.")
else:
    raise FileNotFoundError(f"Model checkpoint not found at '{MODEL_PATH}'.")

# Dummy Tensors
dummy_spec = torch.randn(1, 1, 10, 10, dtype=torch.float32)
dummy_tab = torch.randn(1, 6, dtype=torch.float32)

# Export to ONNX with Dynamic Axes & Opset 17
torch.onnx.export(
    model,
    (dummy_spec, dummy_tab),
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['spec_input', 'tabular_input'],
    output_names=['output'],
    dynamic_axes={
        'spec_input': {0: 'batch_size'},
        'tabular_input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

onnx_model = onnx.load(ONNX_OUTPUT_PATH)
onnx.checker.check_model(onnx_model)
print(f"[SUCCESS] ONNX model exported and validated at: {ONNX_OUTPUT_PATH}")

# Dynamic INT8 Quantization
quantized_model = torch.ao.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), QUANT_MODEL_PATH)
print(f"[SUCCESS] Dynamic INT8 Quantized model saved to: {QUANT_MODEL_PATH}")


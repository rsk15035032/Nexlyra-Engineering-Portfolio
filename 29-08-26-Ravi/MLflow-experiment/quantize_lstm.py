import os

import torch
import torch.nn as nn

from src.config import (
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT
)

from src.lstm_model import TBMLSTM


# ============================================================
# Paths
# ============================================================

FLOAT32_PATH = (
    "models/pytorch_lstm_best.pt"
)

INT8_PATH = (
    "models/pytorch_lstm_int8.pt"
)


# ============================================================
# Check model
# ============================================================

if not os.path.exists(
    FLOAT32_PATH
):

    raise FileNotFoundError(
        f"Model not found: {FLOAT32_PATH}"
    )


# ============================================================
# Load state dictionary
# ============================================================

state_dict = torch.load(
    FLOAT32_PATH,
    map_location="cpu"
)


# ============================================================
# Find input size
# ============================================================

input_size = (
    state_dict[
        "lstm.weight_ih_l0"
    ].shape[1]
)


print(
    f"Input size: {input_size}"
)


# ============================================================
# Recreate model
# ============================================================

model = TBMLSTM(

    input_size=input_size,

    hidden_size=
        LSTM_HIDDEN_SIZE,

    num_layers=
        LSTM_NUM_LAYERS,

    dropout=
        LSTM_DROPOUT
)


# ============================================================
# Load trained weights
# ============================================================

model.load_state_dict(
    state_dict
)

model.eval()


# ============================================================
# Float32 size
# ============================================================

float32_size = (

    os.path.getsize(
        FLOAT32_PATH
    )

    / (1024 ** 2)
)


print(
    f"\nFloat32 model size: "
    f"{float32_size:.2f} MB"
)


# ============================================================
# INT8 Quantization
# ============================================================

print(
    "\nApplying INT8 quantization..."
)


quantized_model = (
    torch.quantization.quantize_dynamic(

        model,

        {
            nn.LSTM,
            nn.Linear
        },

        dtype=torch.qint8
    )
)


# ============================================================
# Save INT8
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


torch.save(
    quantized_model.state_dict(),
    INT8_PATH
)


# ============================================================
# INT8 size
# ============================================================

int8_size = (

    os.path.getsize(
        INT8_PATH
    )

    / (1024 ** 2)
)


# ============================================================
# Compression
# ============================================================

compression_ratio = (

    float32_size
    /
    int8_size
)


size_reduction = (

    1
    -
    int8_size
    /
    float32_size

) * 100


# ============================================================
# Results
# ============================================================

print(
    f"INT8 model size: "
    f"{int8_size:.2f} MB"
)


print(
    f"Compression ratio: "
    f"{compression_ratio:.2f}x"
)


print(
    f"Size reduction: "
    f"{size_reduction:.2f}%"
)


print(
    "\n✓ INT8 model saved:"
)

print(
    INT8_PATH
)
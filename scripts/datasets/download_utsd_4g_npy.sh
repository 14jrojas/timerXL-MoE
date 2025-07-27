#!/bin/bash

# Activa modo estricto
set -e

echo "Downloading and converting UTSD-4G to .npy..."

# Asegura que estás en la raíz del proyecto
ROOT_DIR=$(pwd)
OUTPUT_DIR="$ROOT_DIR/dataset/UTSD-4G-npy"

# Crear carpeta de salida si no existe
mkdir -p "$OUTPUT_DIR"

# Ejecuta el script Python para convertir a .npy
python3.11 - <<EOF
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

subset = "UTSD-4G"
output_dir = "$OUTPUT_DIR"

print(f"Downloading {subset} from HuggingFace...")
ds = load_dataset("thuml/UTSD", subset, split="train")

print(f"Saving .npy files to: {output_dir}")
for idx, item in tqdm(enumerate(ds), total=len(ds)):
    series = np.array(item["target"], dtype=np.float32).reshape(-1, 1)
    filename = os.path.join(output_dir, f"{subset}_{idx:06d}.npy")
    np.save(filename, series)
EOF

echo "UTSD-4G saved to: $OUTPUT_DIR"

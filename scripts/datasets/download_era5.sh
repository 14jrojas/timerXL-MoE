#!/bin/bash

ROOT_DIR="./dataset"

# URL de descarga (archivo zip)
URL="https://cloud.tsinghua.edu.cn/f/7fe0b95032c64d39bc4a/?dl=1"
ZIP_FILE="ERA5.zip"

# Crear carpeta raíz si no existe
if [ ! -d "$ROOT_DIR" ]; then
    echo "Creando carpeta raíz: $ROOT_DIR"
    mkdir -p "$ROOT_DIR"
fi

echo "Descargando ERA5..."
wget --no-check-certificate "$URL" -O "$ZIP_FILE"

echo "Descomprimiendo..."
unzip -q "$ZIP_FILE" -d "$ROOT_DIR"

echo "Eliminando archivo zip temporal..."
rm "$ZIP_FILE"

echo "Descarga completada. Archivos extraídos en: $ROOT_DIR"

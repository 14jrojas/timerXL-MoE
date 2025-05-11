#!/bin/bash

ROOT_DIR="./dataset"

# URL de descarga (archivo zip)
URL="https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/?dl=1"
ZIP_FILE="UTSD-full-npy.zip"

# Crear carpeta raíz si no existe
if [ ! -d "$ROOT_DIR" ]; then
    echo "Creando carpeta raíz: $ROOT_DIR"
    mkdir -p "$ROOT_DIR"
fi

echo "Descargando UTSD en formato npy..."
wget --no-check-certificate "$URL" -O "$ZIP_FILE"

echo "Descomprimiendo..."
unzip -q "$ZIP_FILE" -d "$ROOT_DIR"

echo "Eliminando archivo zip temporal..."
rm "$ZIP_FILE"

echo "Descarga completada. Archivos extraídos en: $ROOT_DIR"

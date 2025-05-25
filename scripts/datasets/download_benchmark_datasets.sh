#!/bin/bash

set -e

echo "Descargando benchmark datasets..."

mkdir -p dataset/
wget -nc -O dataset/datasets.zip https://cloud.tsinghua.edu.cn/f/4d83223ad71047e28aec/?dl=1

echo "Descomprimiendo archivo..."
unzip -n dataset/datasets.zip/

echo "Limpieza de archivos temporales..."
rm -f dataset/datasets.zip

echo "Descarga y preparación de benchmark datasets completada."
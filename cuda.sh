#!/bin/bash
VERSION="12.8.0"
IMAGE="docker.io/nvidia/cuda:$VERSION-devel-ubuntu24.04"
TARGET_PATH="/usr/local/cuda/targets/x86_64-linux"
LOCAL_DIR=".cuda"

rm -r "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

CONTAINER_ID=$(podman create "$IMAGE")

podman cp "$CONTAINER_ID:$TARGET_PATH/." "$LOCAL_DIR/"
rm -rf "$LOCAL_DIR/lib/stubs"
podman rm "$CONTAINER_ID"

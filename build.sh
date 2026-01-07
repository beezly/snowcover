#!/bin/bash
set -e

# Build script for SnowCover Docker image

IMAGE_NAME="snowcover"
VERSION=$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)

echo "Building ${IMAGE_NAME}:${VERSION}"

docker build \
    -t "${IMAGE_NAME}:${VERSION}" \
    -t "${IMAGE_NAME}:latest" \
    .

echo ""
echo "Build complete!"
echo "  ${IMAGE_NAME}:${VERSION}"
echo "  ${IMAGE_NAME}:latest"
echo ""
echo "Run with:"
echo "  docker run --rm -e RTSP_URL='rtsp://...' -e MQTT_HOST='...' ${IMAGE_NAME}"
echo ""
echo "Or use docker compose:"
echo "  docker compose up -d"

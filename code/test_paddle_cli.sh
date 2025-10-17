#!/bin/bash
# Test script for PaddleOCR CLI on local images

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to test image (first PNG in data/images directory)
IMAGE_DIR="$SCRIPT_DIR/data/images"
TEST_IMAGE=$(ls "$IMAGE_DIR"/*.png | head -n 1)

if [ -z "$TEST_IMAGE" ]; then
    echo "❌ No PNG images found in $IMAGE_DIR"
    exit 1
fi

echo "🔍 Testing PaddleOCR CLI on: $(basename $TEST_IMAGE)"
echo "📁 Full path: $TEST_IMAGE"
echo ""

# Run PaddleOCR CLI
echo "🚀 Running: paddleocr doc_parser -i $TEST_IMAGE"
echo ""

paddleocr doc_parser -i "$TEST_IMAGE"

# print the output
echo "Output:"
echo "$OUTPUT"

echo ""
echo "✅ Test completed!"


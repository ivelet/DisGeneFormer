#!/bin/bash
# download_data.sh - Download dataset for GeneNet/DiseaseNet project
#
# This script downloads and extracts the required dataset from Zenodo.
#
# Usage:
#   ./download_data.sh              # Download and extract dataset
#   ./download_data.sh --keep-zip   # Keep zip file after extraction

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_URL="https://zenodo.org/records/18346612/files/data.zip?download=1"
OUTPUT_FILE="data.zip"
KEEP_ZIP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-zip)
            KEEP_ZIP=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--keep-zip]"
            echo ""
            echo "Options:"
            echo "  --keep-zip    Keep downloaded zip file after extraction"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"
for cmd in wget unzip; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd is not installed${NC}"
        echo -e "${YELLOW}Install with: sudo apt-get install $cmd${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ All prerequisites installed${NC}\n"

# Download dataset
echo -e "${BLUE}Downloading datasets...${NC}"
if wget -q --show-progress -O "$OUTPUT_FILE" "$DATA_URL"; then
    echo -e "${GREEN}✓ Download complete${NC}\n"
else
    echo -e "${RED}✗ Download failed${NC}"
    echo -e "${YELLOW}Please check your internet connection and try again.${NC}"
    exit 1
fi

# Extract dataset
echo -e "${BLUE}Extracting datasets...${NC}"
if unzip -oq "$OUTPUT_FILE"; then
    echo -e "${GREEN}✓ Extraction complete${NC}\n"
else
    echo -e "${RED}✗ Extraction failed${NC}"
    exit 1
fi

# Cleanup zip file unless --keep-zip is set
if [ "$KEEP_ZIP" = false ]; then
    rm -f "$OUTPUT_FILE"
    echo -e "${GREEN}✓ Cleaned up zip file${NC}"
else
    echo -e "${YELLOW}ℹ Kept zip file: $OUTPUT_FILE${NC}"
fi

# Summary
echo -e "${GREEN}✓ Dataset downloaded and extracted successfully!${NC}"
echo -e "${GREEN}Setup complete!${NC}"
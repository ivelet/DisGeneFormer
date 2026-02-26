#!/bin/bash
# download_model_comparison_results.sh - Download and merge model comparison results

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
MODEL_COMPARISON_URL="https://zenodo.org/records/18787455/files/model_comparison.zip?download=1"
ZIP_FILE="model_comparison.zip"
KEEP_ZIP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-zip) KEEP_ZIP=true; shift ;;
        -h|--help) echo "Usage: $0 [--keep-zip]"; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"
for cmd in wget unzip; do
    command -v $cmd &> /dev/null || { echo -e "${RED}$cmd not installed${NC}"; exit 1; }
done
echo -e "${GREEN}✓ Prerequisites OK${NC}\n"

# Download
echo -e "${BLUE}Downloading model_comparison.zip...${NC}"
wget -q --show-progress -O "$ZIP_FILE" "$MODEL_COMPARISON_URL"
echo -e "${GREEN}✓ Downloaded${NC}\n"

# Unzip directly (auto-creates/merges into results/model_comparison/)
echo -e "${BLUE}Extracting and merging into results/model_comparison/...${NC}"
unzip -oq "$ZIP_FILE"
echo -e "${GREEN}✓ Extracted and merged${NC}\n"

# Delete zip unless --keep-zip
if [[ "$KEEP_ZIP" == "false" ]]; then
    rm -f "$ZIP_FILE"
    echo -e "${GREEN}✓ Removed $ZIP_FILE${NC}"
else
    echo -e "${GREEN}ℹ Kept $ZIP_FILE${NC}"
fi

# Summary
echo -e "\n${GREEN}✓ Done! Methods merged into results/model_comparison/${NC}"
echo -e "\n${BLUE}Methods available:${NC}"
ls -1 results/model_comparison/ 2>/dev/null | sed 's/^/  • /' || echo "  (directory not found)"
#!/bin/bash
# download_all_results.sh - Download full results directory (without models)
#
# This script downloads and extracts the complete results directory
# (excluding .ptm model files, gene_net, and disease_net folders).
#
# Usage:
#   ./download_all_results.sh              # Download and extract results
#   ./download_all_results.sh --keep-zip   # Keep zip file after extraction

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - Fill in the actual Zenodo URL
RESULTS_URL="https://zenodo.org/records/18787037/files/results.zip?download=1"
OUTPUT_FILE="results.zip"
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
            echo "Downloads complete results directory (without .ptm model files)"
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

# Check if URL is configured
if [[ -z "$RESULTS_URL" ]]; then
    echo -e "${RED}Error: RESULTS_URL is not configured${NC}"
    echo -e "${YELLOW}Edit this script and set RESULTS_URL to your Zenodo download link${NC}"
    exit 1
fi

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

# Warn if results directory already exists
if [[ -d "results" ]]; then
    echo -e "${YELLOW}⚠ Warning: results/ directory already exists${NC}"
    echo -e "${YELLOW}   Files in the archive will overwrite matching local files${NC}"
    echo -e "${YELLOW}   (.ptm models and other files not in archive will be preserved)${NC}"
    read -p "   Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Aborted${NC}"
        exit 0
    fi
    echo ""
fi

# Download results
echo -e "${BLUE}Downloading results archive...${NC}"
echo -e "${BLUE}Note: This may take a while (excludes .ptm model files)${NC}\n"
if wget -q --show-progress -O "$OUTPUT_FILE" "$RESULTS_URL"; then
    echo -e "${GREEN}✓ Download complete${NC}\n"
else
    echo -e "${RED}✗ Download failed${NC}"
    echo -e "${YELLOW}Please check your internet connection and try again.${NC}"
    exit 1
fi

# Extract results
echo -e "${BLUE}Extracting results...${NC}"
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

# Clean up unwanted directories that shouldn't be in the archive
echo -e "\n${BLUE}Cleaning up unwanted directories...${NC}"
UNWANTED_DIRS=(
    "results/humannet_comparison/humannet_fn_v2_random_negatives"
    "results/humannet_comparison/humannet_fn_v3_random_negatives"
    "results/humannet_comparison/humannet_xc_v3_random_negatives"
    "results/humannet_comparison/humannet_xc_v3_filtered_random_negatives"
    "results/humannet_comparison/humannet_xn_v2_random_negatives"
    "results/best_model_copy"
    "results/gene_net_xc_v3_feature_removal"
)

REMOVED_COUNT=0
for dir in "${UNWANTED_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        echo -e "  ${YELLOW}• Removing: $dir${NC}"
        rm -rf "$dir"
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
    fi
done

if [[ $REMOVED_COUNT -gt 0 ]]; then
    echo -e "${GREEN}✓ Removed $REMOVED_COUNT unwanted directory/directories${NC}"
else
    echo -e "${GREEN}✓ No unwanted directories found${NC}"
fi

# Summary
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Results downloaded and extracted successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Location: ./results/${NC}"
echo -e "${YELLOW}Note: Model files (.ptm) are excluded from this archive${NC}"
echo -e "${GREEN}Setup complete!${NC}"
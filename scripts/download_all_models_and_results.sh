#!/bin/bash
# download_all_models_and_results.sh - Download experimental results for GeneNet/DiseaseNet project
#
# This script downloads and extracts experimental result archives from Zenodo.
#
# Usage:
#   ./download_all_models_and_results.sh              # Download and extract all results
#   ./download_all_models_and_results.sh --keep-zip   # Keep zip files after extraction
#   ./download_all_models_and_results.sh --only best_model  # Download specific result only

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DOWNLOAD_ALL_URL = "https://zenodo.org/api/records/18791945/files-archive"

# ============================================================================
# RESULT ARCHIVE URLs - Fill in the actual Zenodo URLs here
# ============================================================================
declare -A RESULT_URLS=(
    ["best_model"]="https://zenodo.org/records/18791945/files/best_model.zip?download=1"
    ["best_model_xc_v3"]="https://zenodo.org/records/18791945/files/best_model_xc_v3.zip?download=1"
    ["disease_net_feature_removal"]="https://zenodo.org/records/18791945/files/disease_net_feature_removal.zip?download=1"
    ["gene_net_feature_removal"]="https://zenodo.org/records/18791945/files/gene_net_feature_removal.zip?download=1"
    ["humannet_comparison"]="https://zenodo.org/records/18791945/files/humannet_comparison.zip?download=1"
    ["model_comparison"]="https://zenodo.org/records/18791945/files/model_comparison.zip?download=1"
    ["negative_comparison"]="https://zenodo.org/records/18791945/files/negative_comparison.zip?download=1"
)

# Result archive names
declare -A RESULT_FILES=(
    ["best_model"]="best_model.zip"
    ["best_model_xc_v3"]="best_model_xc_v3.zip"
    ["disease_net_feature_removal"]="disease_net_feature_removal.zip"
    ["gene_net_feature_removal"]="gene_net_feature_removal.zip"
    ["humannet_comparison"]="humannet_comparison.zip"
    ["model_comparison"]="model_comparison.zip"
    ["negative_comparison"]="negative_comparison.zip"
)

# Configuration
RESULTS_DIR="results"
KEEP_ZIP=false
DOWNLOAD_ONLY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-zip)
            KEEP_ZIP=true
            shift
            ;;
        --only)
            if [[ -z "${2-}" ]]; then
                echo -e "${RED}Error: --only requires an argument${NC}"
                exit 1
            fi
            DOWNLOAD_ONLY="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --keep-zip         Keep downloaded zip files after extraction"
            echo "  --only <name>      Download only specific result archive"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Available archives:"
            for key in "${!RESULT_FILES[@]}"; do
                echo "  - $key"
            done | sort
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

# Create results directory
echo -e "${BLUE}Creating results directory...${NC}"
mkdir -p "$RESULTS_DIR"
echo -e "${GREEN}✓ Results directory ready${NC}\n"

# Function to download and extract a single result archive
download_and_extract() {
    local name=$1
    local url=$2
    local filename=$3
    
    # Skip if URL is empty
    if [[ -z "$url" ]]; then
        echo -e "${YELLOW}⚠ Skipping $name (URL not configured)${NC}"
        return 0
    fi
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Processing: $name${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Download
    echo -e "${BLUE}Downloading $filename...${NC}"
    if wget -q --show-progress -O "$filename" "$url"; then
        echo -e "${GREEN}✓ Download complete${NC}\n"
    else
        echo -e "${RED}✗ Download failed for $name${NC}"
        echo -e "${YELLOW}Continuing with next archive...${NC}\n"
        return 1
    fi
    
    # Extract to results directory
    echo -e "${BLUE}Extracting $filename to $RESULTS_DIR/...${NC}"
    if unzip -oq "$filename" -d "$RESULTS_DIR"; then
        echo -e "${GREEN}✓ Extraction complete${NC}\n"
    else
        echo -e "${RED}✗ Extraction failed for $name${NC}"
        return 1
    fi
    
    # Cleanup zip file unless --keep-zip is set
    if [ "$KEEP_ZIP" = false ]; then
        rm -f "$filename"
        echo -e "${GREEN}✓ Cleaned up $filename${NC}\n"
    else
        echo -e "${YELLOW}ℹ Kept zip file: $filename${NC}\n"
    fi
    
    return 0
}

# Determine which archives to download
if [[ -n "$DOWNLOAD_ONLY" ]]; then
    # Validate the archive name
    if [[ ! -v "RESULT_FILES[$DOWNLOAD_ONLY]" ]]; then
        echo -e "${RED}Error: Unknown archive '$DOWNLOAD_ONLY'${NC}"
        echo -e "${YELLOW}Available archives:${NC}"
        for key in "${!RESULT_FILES[@]}"; do
            echo "  - $key"
        done | sort
        exit 1
    fi
    
    echo -e "${BLUE}Downloading only: $DOWNLOAD_ONLY${NC}\n"
    ARCHIVES_TO_DOWNLOAD=("$DOWNLOAD_ONLY")
else
    echo -e "${BLUE}Downloading all result archives${NC}\n"
    ARCHIVES_TO_DOWNLOAD=("${!RESULT_FILES[@]}")
fi

# Download and extract archives
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

for name in "${ARCHIVES_TO_DOWNLOAD[@]}"; do
    url="${RESULT_URLS[$name]}"
    filename="${RESULT_FILES[$name]}"
    
    if [[ -z "$url" ]]; then
        ((SKIP_COUNT++))
        echo -e "${YELLOW}⚠ Skipping $name (URL not configured)${NC}\n"
        continue
    fi
    
    if download_and_extract "$name" "$url" "$filename"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

# Summary
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Download Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Successful: $SUCCESS_COUNT${NC}"
if [[ $FAIL_COUNT -gt 0 ]]; then
    echo -e "${RED}✗ Failed: $FAIL_COUNT${NC}"
fi
if [[ $SKIP_COUNT -gt 0 ]]; then
    echo -e "${YELLOW}⚠ Skipped: $SKIP_COUNT${NC}"
fi

if [[ $SUCCESS_COUNT -gt 0 ]]; then
    echo -e "\n${GREEN}✓ Results extracted to: $RESULTS_DIR/${NC}"
fi

if [[ $SKIP_COUNT -gt 0 ]]; then
    echo -e "\n${YELLOW}Note: Some archives were skipped because URLs are not configured.${NC}"
    echo -e "${YELLOW}Edit this script and fill in the RESULT_URLS array with actual URLs.${NC}"
fi

if [[ $FAIL_COUNT -eq 0 && $SKIP_COUNT -eq 0 ]]; then
    echo -e "\n${GREEN}✓ Setup complete!${NC}"
    exit 0
elif [[ $SUCCESS_COUNT -gt 0 ]]; then
    echo -e "\n${YELLOW}⚠ Setup partially complete${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Setup failed${NC}"
    exit 1
fi
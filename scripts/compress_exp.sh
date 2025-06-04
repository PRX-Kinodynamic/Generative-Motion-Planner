#!/bin/bash

# Exit on any error
set -e

# Create experiments directory if it doesn't exist
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
EXPERIMENTS_DIR="${SCRIPT_DIR}/experiments"
mkdir -p "${EXPERIMENTS_DIR}"

# Function to copy specific files from an experiment path
copy_experiment_files() {
    local exp_path="$1"
    
    # Make sure the path exists
    if [ ! -d "${exp_path}" ]; then
        echo "Error: ${exp_path} is not a valid directory"
        return 1
    fi
    
    # Get the full path
    local full_path="$(realpath "${exp_path}")"
    
    # Extract the experiment name/structure (last part of the path)
    local exp_name="$(basename "${full_path}")"
    local parent_dir="$(dirname "${full_path}")"
    local parent_name="$(basename "${parent_dir}")"
    local dest_dir="${EXPERIMENTS_DIR}/${parent_name}/${exp_name}"
    
    # Create the destination directory
    mkdir -p "${dest_dir}"
    
    # Copy the specific files and directories
    echo "Copying files from ${exp_path} to ${dest_dir}..."
    
    # Copy directories if they exist
    if [ -d "${full_path}/final_states" ]; then
        cp -r "${full_path}/final_states" "${dest_dir}/"
    fi
    
    if [ -d "${full_path}/results" ]; then
        cp -r "${full_path}/results" "${dest_dir}/"
    fi
    
    # Copy individual files if they exist
    for file in args.json best.pt train_losses.txt val_losses.txt; do
        if [ -f "${full_path}/${file}" ]; then
            cp "${full_path}/${file}" "${dest_dir}/"
        fi
    done
    
    echo "Done copying ${exp_path}"
    
    # Compress the destination directory
    local parent_dest_dir="${EXPERIMENTS_DIR}/${parent_name}"
    local archive_name="${parent_dest_dir}/${exp_name}.tar.gz"
    
    echo "Compressing ${dest_dir} to ${archive_name}..."
    tar -czf "${archive_name}" -C "${parent_dest_dir}" "${exp_name}"
    
    # Remove the uncompressed directory after compression
    if [ $? -eq 0 ]; then
        echo "Removing ${dest_dir}..."
        rm -rf "${dest_dir}"
        echo "Compression complete: ${archive_name}"
    else
        echo "Compression failed, keeping uncompressed directory"
    fi
}

# Print usage if no arguments are provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_path1> <experiment_path2> ..."
    echo "Copies specific files from experiment directories to ${EXPERIMENTS_DIR},"
    echo "compresses them to tar.gz archives, and removes the uncompressed directories."
    exit 1
fi

# Process each experiment path
for exp_path in "$@"; do
    copy_experiment_files "${exp_path}"
done

echo "All experiments copied and compressed to ${EXPERIMENTS_DIR}"

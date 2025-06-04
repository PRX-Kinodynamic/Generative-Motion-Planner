#!/bin/bash

# Recursively find all directories named "generated_trajectories" and rename them to "final_states"
find /common/home/st1122/Projects/genMoPlan/experiments -type d -name "generated_trajectories" -exec sh -c '
    old_path="$1"
    new_path="${old_path%/*}/final_states"
    echo "Renaming: $old_path → $new_path"
    mv "$old_path" "$new_path" || echo "Failed to rename $old_path"
' sh {} \;

echo "Directory renaming completed."

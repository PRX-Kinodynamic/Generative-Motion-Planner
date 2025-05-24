import os
import glob

def has_best_pt(directory):
    return os.path.exists(os.path.join(directory, "best.pt"))


def expand_model_paths(model_paths):
    expanded_paths = []
    
    for path in model_paths:
        if "*" in path:
            # Handle wildcard paths
            # Use glob.glob directly with the path containing the wildcard
            all_dirs = glob.glob(path)
            
            # Filter directories that contain best.pt
            valid_dirs = [d for d in all_dirs if os.path.isdir(d) and has_best_pt(d)]
            
            # Add valid directories to expanded paths
            expanded_paths.extend(valid_dirs)
        else:
            # Direct path without wildcard
            if os.path.exists(path) and has_best_pt(path):
                expanded_paths.append(path)
            else:
                print(f"Warning: Path {path} does not exist or does not contain best.pt")

    print(f"Found {len(expanded_paths)} valid model paths:")
    for model_path in expanded_paths:
        print(f"  - {model_path}")

    input("Press Enter to continue...")
    
    return expanded_paths









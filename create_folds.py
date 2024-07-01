import os
import shutil
import random

def create_folds(healthy_dir, unhealthy_dir, output_dir, n_folds=10):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List files in both directories
    healthy_files = os.listdir(healthy_dir)
    unhealthy_files = os.listdir(unhealthy_dir)

    # Shuffle the files to ensure randomness
    random.shuffle(healthy_files)
    random.shuffle(unhealthy_files)

    # Create lists to hold the files for each fold
    healthy_folds = [[] for _ in range(n_folds)]
    unhealthy_folds = [[] for _ in range(n_folds)]

    # Distribute files into folds
    for i, file in enumerate(healthy_files):
        healthy_folds[i % n_folds].append(file)
    for i, file in enumerate(unhealthy_files):
        unhealthy_folds[i % n_folds].append(file)

    # Create folders for each fold
    for fold_idx in range(n_folds):
        fold_dir = os.path.join(output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'healthy'), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'unhealthy'), exist_ok=True)

        # Copy healthy files to the fold
        for file in healthy_folds[fold_idx]:
            src = os.path.join(healthy_dir, file)
            dst = os.path.join(fold_dir, 'healthy', file)
            shutil.copyfile(src, dst)

        # Copy unhealthy files to the fold
        for file in unhealthy_folds[fold_idx]:
            src = os.path.join(unhealthy_dir, file)
            dst = os.path.join(fold_dir, 'unhealthy', file)
            shutil.copyfile(src, dst)

        print(f"Fold {fold_idx + 1} created with {len(healthy_folds[fold_idx])} healthy and {len(unhealthy_folds[fold_idx])} unhealthy recordings.")

# Example usage
healthy_dir = 'trimmed_files/saarbruecken_m_n'  # Replace with your directory path for healthy recordings
unhealthy_dir = 'trimmed_files/saarbruecken_m_p'  # Replace with your directory path for unhealthy recordings
output_dir = 'folds_dataset'  # Replace with your desired output directory path

create_folds(healthy_dir, unhealthy_dir, output_dir)
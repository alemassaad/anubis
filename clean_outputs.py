# clean_outputs.py

import os
import shutil
import sys

def get_directory_summary(directory):
    """
    Compute a summary of the directory contents.
    
    Parameters
    ----------
    directory : str
        Path to the directory.
        
    Returns
    -------
    tuple
        (file_count, total_size_bytes)
    """
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        file_count += len(filenames)
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return file_count, total_size

def format_size(bytes_size):
    """
    Format a byte size into a human-readable string.
    
    Parameters
    ----------
    bytes_size : int
        Size in bytes.
        
    Returns
    -------
    str
        Human-readable size.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}PB"

def clean_outputs(output_dir='outputs'):
    """
    Remove the entire output directory and all its contents after confirmation.
    
    This function calculates the number of files and total size of data
    in the output directory, displays the summary, and then asks for confirmation.
    If confirmed, it deletes the directory recursively.
    
    Parameters
    ----------
    output_dir : str, optional
        The directory to remove. Default is 'outputs'.
        
    Returns
    -------
    None
    """
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' does not exist. Nothing to clean.")
        return

    file_count, total_size = get_directory_summary(output_dir)
    print(f"The directory '{output_dir}' contains {file_count} files "
          f"with a total size of {format_size(total_size)}.")

    confirm = input("Are you sure you want to completely remove this directory? (Y/N): ")
    if confirm.lower() == 'y':
        try:
            shutil.rmtree(output_dir)
            print(f"Output directory '{output_dir}' and all its contents have been removed.")
        except Exception as e:
            print(f"An error occurred while deleting '{output_dir}': {e}")
            sys.exit(1)
    else:
        print("Operation cancelled. No files were deleted.")

if __name__ == "__main__":
    clean_outputs()

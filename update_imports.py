#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Import updater script for Queue Prioritization package.

This script updates imports and file paths in the codebase.
"""

import os
import re
import sys

def update_imports_in_file(file_path):
    """Update imports in a single file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update algorithm imports
    content = re.sub(
        r'from (Alg\d+_[a-z_]+) import',
        r'from queue_prioritization.algorithms.\1 import',
        content
    )
    
    # Update relative imports to distributions
    content = re.sub(
        r'from distributions\.([a-z_]+) import',
        r'from queue_prioritization.distributions.\1 import',
        content
    )
    
    # Update relative imports to experiment
    content = re.sub(
        r'from experiment\.([a-z_]+) import',
        r'from queue_prioritization.experiment.\1 import',
        content
    )
    
    # Update relative imports to helpers
    content = re.sub(
        r'from helpers\.([a-z_]+) import',
        r'from queue_prioritization.helpers.\1 import',
        content
    )
    
    # Update file paths to use EXPERIMENTS_DIR
    content = re.sub(
        r'(["\']\s*)results/([^"\']*[\"\'])',
        r'\1" + EXPERIMENTS_DIR + "/\2',
        content
    )
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.write(content)

def update_all_imports(src_dir):
    """Update imports in all Python files in the src directory."""
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Updating imports in {file_path}")
                update_imports_in_file(file_path)

if __name__ == "__main__":
    src_dir = "src/queue_prioritization"
    if not os.path.exists(src_dir):
        print(f"Error: Directory {src_dir} does not exist")
        sys.exit(1)
    
    update_all_imports(src_dir)
    print("Import update complete!") 
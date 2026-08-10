"""Optional connected-component post-processing for 3D segmentation predictions.

This creates cleaned versions of predictions without modifying raw model outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import label


def filter_connected_components(
    pred_data: np.ndarray,
    min_voxel_count: int = 100,
    keep_largest_per_class: bool = True,
    class_ids: tuple[int, ...] = (1, 2, 3)
) -> np.ndarray:
    """
    Filter small disconnected components from segmentation.
    
    Args:
        pred_data: Prediction volume [D, H, W] with class labels
        min_voxel_count: Minimum voxel count for a component to be kept
        keep_largest_per_class: If True, keep only the largest component per class
        class_ids: Classes to filter (default: tumor classes 1,2,3)
    
    Returns:
        Filtered prediction volume with same shape as input
    """
    pred_clean = pred_data.copy()
    
    for class_id in class_ids:
        # Create binary mask for this class
        class_mask = (pred_data == class_id).astype(np.int16)
        
        if not np.any(class_mask):
            continue
        
        # Find connected components
        labeled_array, num_features = label(class_mask)
        
        if num_features == 0:
            continue
        
        # Count voxels per component
        component_sizes = []
        for component_id in range(1, num_features + 1):
            component_mask = (labeled_array == component_id)
            voxel_count = np.sum(component_mask)
            component_sizes.append((component_id, voxel_count))
        
        # Sort by size (largest first)
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        if keep_largest_per_class:
            # Keep only the largest component for this class
            if component_sizes:
                largest_id = component_sizes[0][0]
                keep_mask = (labeled_array == largest_id)
                
                # Remove all other components of this class
                remove_mask = (labeled_array != largest_id) & (labeled_array > 0)
                pred_clean[remove_mask] = 0
        else:
            # Keep components above minimum size threshold
            for component_id, voxel_count in component_sizes:
                if voxel_count < min_voxel_count:
                    remove_mask = (labeled_array == component_id)
                    pred_clean[remove_mask] = 0
    
    return pred_clean


def postprocess_nifti(
    input_path: Path,
    output_path: Path,
    min_voxel_count: int = 100,
    keep_largest_per_class: bool = True,
    class_ids: tuple[int, ...] = (1, 2, 3)
) -> None:
    """Apply connected-component filtering to a NIfTI file."""
    # Load prediction
    img = nib.as_closest_canonical(nib.load(str(input_path)))
    pred_data = img.get_fdata()
    
    # Convert to canonical orientation [D,H,W]
    pred_data = np.transpose(pred_data, (2, 0, 1))
    pred_data = np.rint(pred_data).astype(np.int16)
    
    # Apply filtering
    pred_clean = filter_connected_components(
        pred_data,
        min_voxel_count=min_voxel_count,
        keep_largest_per_class=keep_largest_per_class,
        class_ids=class_ids
    )
    
    # Convert back to original orientation
    pred_clean = np.transpose(pred_clean, (1, 2, 0))
    
    # Save cleaned version
    out_img = nib.Nifti1Image(pred_clean.astype(np.uint8), affine=img.affine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))
    
    print(f"Post-processed: {input_path.name} -> {output_path.name}")
    print(f"  Min voxel count: {min_voxel_count}")
    print(f"  Keep largest per class: {keep_largest_per_class}")


def main():
    parser = argparse.ArgumentParser(description="Optional connected-component post-processing")
    parser.add_argument("--input", type=str, required=True, help="Input prediction NIfTI")
    parser.add_argument("--output", type=str, required=True, help="Output cleaned NIfTI")
    parser.add_argument("--min_voxel_count", type=int, default=100, help="Minimum voxel count threshold")
    parser.add_argument("--keep_largest", action="store_true", help="Keep only largest component per class")
    parser.add_argument("--no_keep_largest", action="store_false", dest="keep_largest", help="Keep all components above threshold")
    parser.set_defaults(keep_largest=True)
    args = parser.parse_args()
    
    postprocess_nifti(
        input_path=Path(args.input),
        output_path=Path(args.output),
        min_voxel_count=args.min_voxel_count,
        keep_largest_per_class=args.keep_largest
    )


if __name__ == "__main__":
    main()
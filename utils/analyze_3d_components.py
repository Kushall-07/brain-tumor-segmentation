"""Analyze connected components in 3D segmentation predictions to diagnose floating artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import label


def analyze_connected_components(
    pred_path: Path,
    gt_path: Path = None,
    brain_mask_path: Path = None
) -> Dict:
    """Analyze connected components in prediction NIfTI."""
    
    # Load prediction
    pred_img = nib.as_closest_canonical(nib.load(str(pred_path)))
    pred_data = pred_img.get_fdata()
    
    # Get voxel spacing for physical volume calculation
    voxel_spacing = pred_img.header.get_zooms()
    voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    
    # Convert to canonical orientation [D,H,W]
    pred_data = np.transpose(pred_data, (2, 0, 1))
    pred_data = np.rint(pred_data).astype(np.int16)
    
    # Load ground truth if available
    gt_data = None
    if gt_path and gt_path.exists():
        gt_img = nib.as_closest_canonical(nib.load(str(gt_path)))
        gt_data = gt_img.get_fdata()
        gt_data = np.transpose(gt_data, (2, 0, 1))
        gt_data = np.rint(gt_data).astype(np.int16)
    
    # Load brain mask if available
    brain_data = None
    if brain_mask_path and brain_mask_path.exists():
        brain_img = nib.as_closest_canonical(nib.load(str(brain_mask_path)))
        brain_data = brain_img.get_fdata()
        brain_data = np.transpose(brain_data, (2, 0, 1))
        brain_data = brain_data > 0
    
    # Analyze each class
    classes_to_analyze = [1, 2, 3]  # NCR/NET, Edema, ET
    all_components = []
    
    for class_id in classes_to_analyze:
        # Create binary mask for this class
        class_mask = (pred_data == class_id).astype(np.int16)
        
        if not np.any(class_mask):
            continue
        
        # Find connected components
        labeled_array, num_features = label(class_mask)
        
        # Analyze each component
        for component_id in range(1, num_features + 1):
            component_mask = (labeled_array == component_id)
            voxel_count = np.sum(component_mask)
            physical_volume = voxel_count * voxel_volume_mm3
            
            # Find bounding box
            coords = np.argwhere(component_mask)
            if coords.size > 0:
                min_coords = coords.min(axis=0)
                max_coords = coords.max(axis=0)
                centroid = coords.mean(axis=0)
                
                # Check if inside brain (if brain mask available)
                inside_brain = None
                if brain_data is not None:
                    # Sample points inside component
                    sample_points = coords[::max(1, len(coords)//100)]  # sample for efficiency
                    inside_brain = np.any([brain_data[tuple(p)] for p in sample_points])
                
                # Check if present in ground truth
                in_gt = None
                if gt_data is not None:
                    sample_points = coords[::max(1, len(coords)//100)]
                    in_gt = np.any([gt_data[tuple(p)] == class_id for p in sample_points])
                
                component_info = {
                    "class": int(class_id),
                    "component_id": int(component_id),
                    "voxel_count": int(voxel_count),
                    "physical_volume_mm3": float(physical_volume),
                    "bounding_box": {
                        "min": [int(x) for x in min_coords.tolist()],
                        "max": [int(x) for x in max_coords.tolist()]
                    },
                    "centroid": [float(x) for x in centroid.tolist()],
                    "inside_brain": bool(inside_brain) if inside_brain is not None else None,
                    "in_ground_truth": bool(in_gt) if in_gt is not None else None
                }
                
                all_components.append(component_info)
    
    # Find primary component (largest) for each class
    primary_components = {}
    for class_id in classes_to_analyze:
        class_components = [c for c in all_components if c["class"] == class_id]
        if class_components:
            primary = max(class_components, key=lambda x: x["voxel_count"])
            primary_components[class_id] = primary
        else:
            print(f"Warning: No components found for class {class_id}")
    
    # Find disconnected small components
    disconnected = []
    for component in all_components:
        class_id = component["class"]
        if class_id in primary_components:
            primary = primary_components[class_id]
            # Consider disconnected if significantly smaller than primary
            if component["voxel_count"] < primary["voxel_count"] * 0.1:  # less than 10% of primary
                disconnected.append(component)
        else:
            # No primary found, mark as disconnected
            disconnected.append(component)
    
    return {
        "prediction_file": str(pred_path),
        "ground_truth_file": str(gt_path) if gt_path else None,
        "total_components": len(all_components),
        "disconnected_components": len(disconnected),
        "primary_components": primary_components,
        "disconnected_details": disconnected,
        "all_components": all_components,
        "voxel_spacing_mm": [float(x) for x in voxel_spacing],
        "shape": [int(x) for x in pred_data.shape]
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze 3D connected components in predictions")
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction NIfTI")
    parser.add_argument("--gt", type=str, help="Path to ground truth NIfTI (optional)")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()
    
    pred_path = Path(args.pred)
    gt_path = Path(args.gt) if args.gt else None
    
    results = analyze_connected_components(pred_path, gt_path)
    
    # Print summary
    print("=" * 60)
    print("3D COMPONENT ANALYSIS")
    print("=" * 60)
    print(f"Prediction: {pred_path.name}")
    print(f"Total components: {results['total_components']}")
    print(f"Disconnected components: {results['disconnected_components']}")
    
    print("\nPrimary components:")
    print(f"Debug: primary_components dict = {results['primary_components']}")
    if results['primary_components']:
        for class_id, comp in results['primary_components'].items():
            print(f"  Class {class_id}: {comp['voxel_count']} voxels, {comp['physical_volume_mm3']:.2f} mm³")
    else:
        print("  No primary components found")
    
    if results['disconnected_details']:
        print("\nDisconnected components (small components):")
        for comp in results['disconnected_details'][:10]:  # Show first 10
            print(f"  Class {comp['class']}, Component {comp['component_id']}: "
                  f"{comp['voxel_count']} voxels, {comp['physical_volume_mm3']:.2f} mm³")
            print(f"    Bounding box: {comp['bounding_box']}")
            print(f"    Centroid: {comp['centroid']}")
            if comp['inside_brain'] is not None:
                print(f"    Inside brain: {comp['inside_brain']}")
            if comp['in_ground_truth'] is not None:
                print(f"    In ground truth: {comp['in_ground_truth']}")
        
        if len(results['disconnected_details']) > 10:
            print(f"  ... and {len(results['disconnected_details']) - 10} more disconnected components")
    
    # Save detailed results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
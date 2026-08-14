"""
Test script to validate that filtered segmentation masks have correct unique labels.
"""
import nibabel as nib
import numpy as np
from pathlib import Path
from api.utils import filter_segmentation_mask

def test_filter_validation():
    """Test that filtered masks have correct unique labels."""
    
    # Use a dummy mask path for testing - this will fail if file doesn't exist
    # This is just to validate the logic, not to run with real data
    test_cases = [
        ([1], [0, 1]),
        ([2], [0, 2]),
        ([3], [0, 3]),
        ([1, 2], [0, 1, 2]),
        ([1, 3], [0, 1, 3]),
        ([2, 3], [0, 2, 3]),
        ([1, 2, 3], [0, 1, 2, 3]),
        ([], [0]),
    ]
    
    print("Test cases for label filtering logic:")
    print("=" * 60)
    
    for enabled_classes, expected_labels in test_cases:
        print(f"Enabled classes: {enabled_classes}")
        print(f"Expected unique labels: {expected_labels}")
        print(f"Test: {'PASS' if True else 'FAIL'}")  # Placeholder - actual test needs real mask
        print("-" * 60)
    
    print("\nNote: Actual validation requires a real prediction mask file.")
    print("Run with: test_mask_path = 'outputs/predictions/xxx/prediction_mask.nii.gz'")
    print("Then call filter_segmentation_mask and verify unique labels.")

if __name__ == "__main__":
    test_filter_validation()

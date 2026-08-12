"""
Test the backend filter endpoint directly.
"""
import requests
import json

API_BASE = "http://127.0.0.1:8000"

# Test cases
test_cases = [
    ([1], "NCR/NET only"),
    ([2], "Edema only"),
    ([3], "ET only"),
    ([1, 2], "NCR/NET + Edema"),
    ([1, 3], "NCR/NET + ET"),
    ([2, 3], "Edema + ET"),
    ([1, 2, 3], "All classes"),
    ([], "No classes"),
]

# You need to provide an actual prediction mask path
# Replace this with a real path from your predictions
MASK_PATH = "your_prediction_path_here/prediction_mask.nii.gz"

def test_filter_endpoint():
    """Test the filter endpoint with various class combinations."""
    
    print("Testing backend filter endpoint")
    print("=" * 60)
    
    for classes, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Classes: {classes}")
        
        payload = {
            "mask_path": MASK_PATH,
            "classes": classes
        }
        
        try:
            response = requests.post(
                f"{API_BASE}/predict/filter-mask",
                json=payload,
                timeout=30
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Request failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("\nNote: You must replace MASK_PATH with an actual prediction path")
    print("from outputs/predictions/ directory.")

if __name__ == "__main__":
    test_filter_endpoint()

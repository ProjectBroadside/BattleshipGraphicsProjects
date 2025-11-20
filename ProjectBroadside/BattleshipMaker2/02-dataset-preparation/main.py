
import argparse
from validator import DatasetValidator

def main():
    """Main entry point for the dataset validation CLI."""
    parser = argparse.ArgumentParser(description="Validate an image dataset for 3DGS training.")
    parser.add_argument("dataset_path", type=str, help="Path to the directory containing the image dataset.")
    
    parser.add_argument("--run-reconstruction", action="store_true", help="Run the 3D reconstruction preview.")
    
    args = parser.parse_args()
    
    try:
        validator = DatasetValidator(args.dataset_path, args.run_reconstruction)
        report = validator.validate()
        if report:
            validator.print_report(report)
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

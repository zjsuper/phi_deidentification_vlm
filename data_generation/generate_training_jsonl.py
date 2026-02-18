#!/usr/bin/env python3


import os
import json
import glob
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
from tqdm import tqdm


class SalientQTrainingDataGenerator:


    def __init__(self, random_seed):
        random.seed(random_seed)

        self.phi_category_mapping = {
            "Name": "name", "DOB": "DOB", "SSN": "SSN",
            "Address": "address", "Phone": "phone", "Email": "email",
            "MRN": "medical record numbers", "Sex": "sex",
            "name": "name", "dob": "DOB", "ssn": "SSN",
            "address": "address", "phone": "phone", "email": "email",
            "mrn": "medical record numbers", "sex": "sex",
        }

        self.prompts = [
            """Analyze this medical image carefully. Locate any visible PHI (Protected Health Information) in this image and output the type and corresponding bounding box in JSON format. Notice there may be multiple PHI elements in the image. If no PHI is found, indicate this in the JSON response.

PHI text can appear in UPPERCASE, lowercase, or Mixed Case. Common formats include:
- Single line with labels: "Name: John Smith, DOB: 1985-03-15" or "NAME: JOHN SMITH, DOB: 1985-03-15"
- Single line values only: "John Smith, 1985-03-15" or "JOHN SMITH, 1985-03-15"
- Multiple lines with labels: "Name: Sarah Johnson\\nAddress: 456 Oak Ave\\nPhone: (555) 123-4567"
- Multiple lines values only: "Sarah Johnson\\n456 Oak Ave\\n(555) 123-4567"

Examples:
- If PHI found: {"phi_found": true, "phi_elements": [{"type": "name", "bbox_2d": [120, 50, 280, 75]}, {"type": "DOB", "bbox_2d": [300, 50, 450, 75]}]}
- If no PHI: {"phi_found": false, "phi_elements": []}

Note: Bounding boxes use [x1, y1, x2, y2] format (top-left to bottom-right coordinates)."""
        ]

        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2

    def split_data(self, data):

        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        total_samples = len(shuffled_data)
        train_end = int(total_samples * self.train_ratio)
        val_end = train_end + int(total_samples * self.val_ratio)

        train_data = shuffled_data[:train_end]
        val_data = shuffled_data[train_end:val_end]
        test_data = shuffled_data[val_end:]

        return train_data, val_data, test_data

    def load_coordinate_files(self, coordinates_dir):
        coord_files = glob.glob(os.path.join(coordinates_dir, "sample_*_*.json"))

        if not coord_files:
            raise FileNotFoundError()

        all_data = []
        failed_files = []

        for coord_file in tqdm(sorted(coord_files), desc="Loading coordinates"):
            try:
                with open(coord_file, 'r') as f:
                    data = json.load(f)
                    if 'output_path' in data and 'phi_coordinates' in data and 'mask_path' in data:
                        all_data.append(data)
                    else:
                        failed_files.append((coord_file, "Missing required fields"))
            except Exception as e:
                failed_files.append((coord_file, str(e)))

        return all_data

    def generate_individual_phi_jsonl(self, coordinate_data, output_file, image_base_path):

        skipped = 0
        unknown_categories = set()

        with open(output_file, 'w') as f:
            for data in tqdm(coordinate_data, desc="Processing individual PHI"):
                image_path = data.get('output_path', '')
                mask_path = data.get('mask_path', '')

                if not image_path:
                    skipped += 1
                    continue

                if image_base_path:
                    try:
                        image_path = os.path.relpath(image_path, image_base_path)
                        mask_path = os.path.relpath(mask_path, image_base_path) if mask_path else ''
                    except ValueError:
                        pass

                individual_phi = []
                for phi_coord in data.get('phi_coordinates', []):
                    phi_category = phi_coord.get('phi_category')

                    if phi_category != 'overall_phi':
                        standardized_category = self.phi_category_mapping.get(phi_category)
                        if standardized_category is None:
                            unknown_categories.add(phi_category)
                            standardized_category = phi_category.lower()

                        abs_coords = phi_coord['coordinates']['absolute']
                        bbox_2d = [abs_coords['x1'], abs_coords['y1'], abs_coords['x2'], abs_coords['y2']]
                        individual_phi.append({"type": standardized_category, "bbox_2d": bbox_2d})

                response = {
                    "phi_found": bool(individual_phi),
                    "phi_elements": individual_phi,
                }

                prompt_idx = data.get('image_id', 0) % len(self.prompts)

                training_entry = {
                    "image": image_path,
                    "mask": mask_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{self.prompts[prompt_idx]}"},
                        {"from": "gpt", "value": json.dumps(response, separators=(',', ':'))},
                    ],
                }

                f.write(json.dumps(training_entry, separators=(',', ':')) + '\n')


    def generate_training_data(self,coordinates_dir,output_dir,image_base_path,max_samples):

        os.makedirs(output_dir, exist_ok=True)
        coordinate_data = self.load_coordinate_files(coordinates_dir)
        if max_samples and max_samples < len(coordinate_data):
            coordinate_data = coordinate_data[:max_samples]

        train_data, val_data, test_data = self.split_data(coordinate_data)

        self.generate_individual_phi_jsonl(train_data, os.path.join(output_dir, "salient_q_train.jsonl"), image_base_path)
        self.generate_individual_phi_jsonl(val_data, os.path.join(output_dir, "salient_q_val.jsonl"), image_base_path)
        self.generate_individual_phi_jsonl(test_data, os.path.join(output_dir, "salient_q_test.jsonl"), image_base_path)



def main():
    parser = argparse.ArgumentParser(description="Generate JSONL training data")
    parser.add_argument("--coordinates-dir", "-c", required=True)
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--image-base-path", "-b", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()

    generator = SalientQTrainingDataGenerator(random_seed=args.random_seed)
    generator.generate_training_data(
        coordinates_dir=args.coordinates_dir,
        output_dir=args.output_dir,
        image_base_path=args.image_base_path,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

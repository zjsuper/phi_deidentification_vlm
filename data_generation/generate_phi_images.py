#!/usr/bin/env python3

import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from data_generation.phi_generator_utils import IntegratedPHIGenerator, create_mask_from_text_image, extract_canny_edges
from data_generation.phi_realistic_degradation import RealisticPHIGenerator


class SalientQDataGenerator:

    def __init__(
        self,output_dir, phi_pairs_path, seed,verbose):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "coordinates").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.phi_generator = IntegratedPHIGenerator(seed=seed)
        self.realistic_generator = RealisticPHIGenerator(verbose=verbose)

        if phi_pairs_path:
            self.pregenerated_pairs = self.load_pregenerated_phi_pairs(phi_pairs_path)
        else:
            self.pregenerated_pairs = None

    def load_pregenerated_phi_pairs(self, phi_pairs_path):

        with open(phi_pairs_path, 'r') as f:
            phi_pairs = json.load(f)
        return phi_pairs

    def load_xray_images(self, xray_dir):

        xray_path = Path(xray_dir)
        image_extensions = ['.png', '.jpg', '.jpeg', '.dcm', '.tif', '.tiff']
        images = []
        for ext in image_extensions:
            images.extend(list(xray_path.glob(f"**/*{ext}")))
        print(f"Found {len(images)} X-ray images")
        return [str(img) for img in images]

    def calculate_coordinates(
        self, x, y, width, height, img_width, img_height):

        x1, y1 = x, y
        x2, y2 = x + width, y + height

        return {
            "yolo": {
                "center_x": (x + width / 2) / img_width,
                "center_y": (y + height / 2) / img_height,
                "width": width / img_width,
                "height": height / img_height,
            },
            "coco": {"x": x, "y": y, "width": width, "height": height},
            "absolute": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "normalized": {
                "x1": x1 / img_width,
                "y1": y1 / img_height,
                "x2": x2 / img_width,
                "y2": y2 / img_height,
            },
        }

    def parse_phi_coordinates(
        self, text_metadata, img_width, img_height):
        phi_data = text_metadata['phi_data']
        phi_positions = text_metadata['phi_positions']
        base_x, base_y = text_metadata['position']
        line_info = text_metadata['line_info']
        font_size = text_metadata['font_size']
        formatted_text = text_metadata['formatted_text']

        lines = formatted_text.split('\n')
        full_text = "\n".join(lines)

        temp_img = Image.new('RGB', (img_width, img_height))
        draw = ImageDraw.Draw(temp_img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        individual_coords = []

        for category, position_info in phi_positions.items():
            phi_start_idx = position_info['start']
            phi_end_idx = position_info['end']
            phi_text = full_text[phi_start_idx:phi_end_idx]

            line_index = 0
            char_count = 0
            position_in_line = phi_start_idx

            for i, line in enumerate(lines):
                if char_count + len(line) >= phi_start_idx:
                    line_index = i
                    position_in_line = phi_start_idx - char_count
                    break
                char_count += len(line) + 1

            line_text = lines[line_index]
            text_before = line_text[:position_in_line]

            if text_before:
                before_bbox = draw.textbbox((0, 0), text_before, font=font)
                before_width = before_bbox[2] - before_bbox[0]
            else:
                before_width = 0

            phi_bbox = draw.textbbox((0, 0), phi_text, font=font)
            phi_width = phi_bbox[2] - phi_bbox[0]
            phi_height = phi_bbox[3] - phi_bbox[1]

            current_y = base_y
            for i in range(line_index):
                current_y += line_info[i]['height']

            phi_x = base_x + before_width
            phi_y = current_y

            coordinates = self.calculate_coordinates(
                phi_x, phi_y, phi_width, phi_height, img_width, img_height
            )

            individual_coords.append({
                'phi_category': category,
                'phi_value': phi_data[category],
                'text_content': phi_text,
                'coordinates': coordinates,
                'font_size': font_size,
                'line_index': line_index,
            })

        return individual_coords

    def create_saliency_mask(
        self,img_size,phi_coordinates,dilation_pixels):
        width, height = img_size
        mask = Image.new('L', (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)

        for phi_coord in phi_coordinates:
            if phi_coord['phi_category'] == 'overall_phi':
                continue

            abs_coords = phi_coord['coordinates']['absolute']
            x1 = max(0, abs_coords['x1'] - dilation_pixels)
            y1 = max(0, abs_coords['y1'] - dilation_pixels)
            x2 = min(width, abs_coords['x2'] + dilation_pixels)
            y2 = min(height, abs_coords['y2'] + dilation_pixels)

            if x1 >= x2 or y1 >= y2:
                continue

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))

            if x1 < x2 and y1 < y2:
                try:
                    mask_draw.rectangle([x1, y1, x2, y2], fill=255)
                except ValueError as e:
                    print(f"Warning: Skipping invalid bbox [{x1}, {y1}, {x2}, {y2}]: {e}")

        return mask

    def generate_single_sample(self, xray_image_path, sample_id, save_intermediates):
        xray_image = Image.open(xray_image_path).convert('RGB')
        img_width, img_height = xray_image.size

        if self.pregenerated_pairs:
            pair_index = sample_id % len(self.pregenerated_pairs)
            phi_pair = self.pregenerated_pairs[pair_index]
            if 'phi_positions' not in phi_pair:
                phi_pair = self.phi_generator.generate_phi_pair_with_positions(sample_id)
        else:
            phi_pair = self.phi_generator.generate_phi_pair_with_positions(sample_id)

        text_image, text_metadata = self.phi_generator.create_phi_text_image(
            phi_pair, xray_image.size, use_aliased_text=True, sample_seed=sample_id,
        )

        generated_image, gen_metadata = self.realistic_generator.generate_burned_in_phi(
            xray_image=xray_image,
            text_image=text_image,
            text_metadata=text_metadata,
            degradation_level='moderate',
            vary_degradation=True,
            seed=sample_id,
        )

        individual_coords = self.parse_phi_coordinates(text_metadata, img_width, img_height)

        bbox = text_metadata['bbox']
        overall_coords = self.calculate_coordinates(
            bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], img_width, img_height,
        )
        individual_coords.append({
            'phi_category': 'overall_phi',
            'phi_value': phi_pair['formatted_text'],
            'text_content': phi_pair['formatted_text'],
            'coordinates': overall_coords,
            'font_size': text_metadata['font_size'],
            'line_index': -1,
        })

        saliency_mask = self.create_saliency_mask(
            img_size=(img_width, img_height),
            phi_coordinates=individual_coords,
            dilation_pixels=3,
        )

        level = gen_metadata.get('degradation_level', 'unknown')
        sample_name = f"sample_{sample_id:06d}_{level}"

        generated_image.save(self.output_dir / "images" / f"{sample_name}.png")
        saliency_mask.save(self.output_dir / "masks" / f"{sample_name}_mask.png")

        if save_intermediates:
            vis_dir = self.output_dir / "visualizations" / sample_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            xray_image.save(vis_dir / "original_xray.png")
            text_image.save(vis_dir / "text_template.png")
            saliency_mask.save(vis_dir / "saliency_mask.png")

            overlay = Image.blend(
                generated_image.convert('RGB'),
                Image.merge('RGB', (
                    saliency_mask,
                    Image.new('L', saliency_mask.size, 0),
                    Image.new('L', saliency_mask.size, 0),
                )),
                alpha=0.3,
            )
            overlay.save(vis_dir / "overlay_visualization.png")

        annotation = {
            'image_id': sample_id,
            'filename': f"{sample_name}.png",
            'mask_filename': f"{sample_name}_mask.png",
            'width': img_width,
            'height': img_height,
            'phi_data': phi_pair['phi_data'],
            'phi_categories': phi_pair['categories'],
            'phi_positions': phi_pair['phi_positions'],
            'formatted_text': phi_pair['formatted_text'],
            'is_single_line': phi_pair['is_single_line'],
            'text_regions': individual_coords,
            'generation_params': gen_metadata,
            'salient_q_metadata': {
                'has_saliency_mask': True,
                'mask_path': f"masks/{sample_name}_mask.png",
                'num_phi_regions': len([c for c in individual_coords if c['phi_category'] != 'overall_phi']),
            },
        }

        with open(self.output_dir / "annotations" / f"{sample_name}.json", 'w') as f:
            json.dump(annotation, f, indent=2)

        with open(self.output_dir / "coordinates" / f"{sample_name}.json", 'w') as f:
            json.dump({
                'image_id': sample_id,
                'image_path': str(xray_image_path),
                'output_path': str(self.output_dir / "images" / f"{sample_name}.png"),
                'mask_path': str(self.output_dir / "masks" / f"{sample_name}_mask.png"),
                'phi_coordinates': individual_coords,
            }, f, indent=2)

        return annotation

    def generate_dataset(self, xray_dir, num_samples):

        xray_images = self.load_xray_images(xray_dir)
        if len(xray_images) == 0:
            return

        all_annotations = []
        with tqdm(total=num_samples, desc="Generating Salient-Q samples") as pbar:
            for sample_id in range(num_samples):
                xray_img_path = xray_images[sample_id % len(xray_images)]
                annotation = self.generate_single_sample(
                    xray_image_path=xray_img_path,
                    sample_id=sample_id,
                    save_intermediates=(sample_id < 10),
                )
                all_annotations.append(annotation)
                pbar.update(1)

        self.save_master_annotations(all_annotations)

    def save_master_annotations(self, annotations):

        master = {
            'info': {
                'description': 'Salient-Q Dataset with Pixel-Level Saliency Masks',
                'version': '1.0',
                'num_samples': len(annotations),
                'generator': 'SalientQDataGenerator',
                'paper': 'KDD 2026 Submission',
            },
            'images': annotations,
        }
        with open(self.output_dir / "annotations_master.json", 'w') as f:
            json.dump(master, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Salient-Q dataset with saliency masks')
    parser.add_argument('--xray_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--phi_pairs', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    generator = SalientQDataGenerator(
        output_dir=args.output_dir,
        phi_pairs_path=args.phi_pairs,
        seed=args.seed,
        verbose=args.verbose,
    )
    generator.generate_dataset(xray_dir=args.xray_dir, num_samples=args.num_samples)


if __name__ == "__main__":
    main()

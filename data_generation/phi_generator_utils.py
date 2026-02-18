# phi_generator_utils.py


import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, List, Optional, Any
import cv2
import urllib.request
import ssl


def extract_canny_edges(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError()
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    return Image.fromarray(dilated_edges).convert('1')


def create_mask_from_text_image(text_image):

    gray = text_image.convert('L')
    mask = gray.point(lambda x: 255 if x > 10 else 0, mode='1')
    return mask


class IntegratedPHIGenerator:
    def __init__(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.colors = {
            "white": (235, 235, 235),
            "off_white": (220, 220, 220),
        }

        self.font_cache_dir = "./fonts_cache"
        os.makedirs(self.font_cache_dir, exist_ok=True)

        self.font_urls = {
            "Roboto-Regular.ttf": "https://github.com/openmaptiles/fonts/raw/master/roboto/Roboto-Regular.ttf",
            "Arimo-Regular.ttf": "https://github.com/arximboldi/sinusoides/raw/master/resources/static/fonts/Arimo-Regular.ttf",
            "Cousine-Regular.ttf": "https://github.com/google/fonts/raw/main/apache/cousine/Cousine-Regular.ttf",
            "IBMPlexMono-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Regular.ttf"
        }

        self.available_fonts = self._download_font_pack()

    def _download_font_pack(self):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        valid_fonts = []

        for name, url in self.font_urls.items():
            path = os.path.join(self.font_cache_dir, name)

            if not os.path.exists(path) or os.path.getsize(path) == 0:
                try:
                    print(f"  Downloading {name}...")
                    with urllib.request.urlopen(url, context=ctx) as response, open(path, 'wb') as out_file:
                        out_file.write(response.read())
                except Exception as e:
                    print(f"  Failed to download {name}: {e}")
                    if os.path.exists(path):
                        os.remove(path)
                    continue

            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        header = f.read(4)
                        if header.startswith(b'<!DO') or header.startswith(b'<htm'):
                            print(f"  {name} appears to be an HTML file (broken link). Deleting.")
                            os.remove(path)
                            continue
                    valid_fonts.append(path)
                except Exception:
                    continue
        return valid_fonts

    def calculate_font_size(self, image_size):
        min_dimension = min(image_size)
        return max(24, int(min_dimension * 0.018))

    def calculate_text_position_avoid_center(self,image_size,text_bbox,):

        W, H = image_size
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        padding = int(min(W, H) * 0.02)

        zones = [
            (0.0, 0.0, 0.3, 0.3),
            (0.7, 0.0, 1.0, 0.3),
            (0.0, 0.7, 0.3, 1.0),
            (0.7, 0.7, 1.0, 1.0),
            (0.0, 0.0, 1.0, 0.15),
        ]

        for _ in range(50):
            zone = random.choice(zones)
            min_x = int(zone[0] * W) + padding
            max_x = int(zone[2] * W) - text_w - padding
            min_y = int(zone[1] * H) + padding
            max_y = int(zone[3] * H) - text_h - padding

            if min_x >= max_x or min_y >= max_y:
                continue
            return random.randint(min_x, max_x), random.randint(min_y, max_y)

        return padding, padding

    def get_adaptive_text_color(self, image, x, y, w, h):
        return self.colors["off_white"]

    def create_phi_text_image(self,phi_pair,image_size,font_path, sample_seed,use_aliased_text):
        if sample_seed is not None:
            random.seed(sample_seed)

        formatted_text = phi_pair['formatted_text']
        is_single_line = phi_pair['is_single_line']
        font_size = self.calculate_font_size(image_size)

        font = None
        if font_path and os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                pass

        if font is None and self.available_fonts:
            chosen_font_path = random.choice(self.available_fonts)
            try:
                font = ImageFont.truetype(chosen_font_path, font_size)
            except Exception as e:
                print(f"Error loading")

        if font is None:
            font = ImageFont.load_default()

        if use_aliased_text:
            img = Image.new('1', image_size, color=0)
            fill_color = 1
        else:
            img = Image.new('RGB', image_size, color=(0, 0, 0))
            fill_color = (255, 255, 255)

        draw = ImageDraw.Draw(img)

        lines = formatted_text.split('\n')
        line_info = []
        max_width = 0
        total_height = 0

        for line in lines:
            try:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                if hasattr(font, 'getmask'):
                    line_height = font.getmask(line).size[1]
                else:
                    line_height = bbox[3] - bbox[1]
            except Exception:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]

            line_height = int(line_height * 1.5)
            line_info.append({'text': line, 'width': line_width, 'height': line_height})
            max_width = max(max_width, line_width)
            total_height += line_height

        box_padding = 10
        text_bbox_size = (max_width + box_padding * 2, total_height + box_padding * 2)
        base_x, base_y = self.calculate_text_position_avoid_center(
            image_size, (0, 0, text_bbox_size[0], text_bbox_size[1])
        )

        draw_x = base_x + box_padding
        current_y = base_y + box_padding

        for line_data in line_info:
            draw.text((draw_x, current_y), line_data['text'], fill=fill_color, font=font)
            current_y += line_data['height']

        overall_bbox = (base_x, base_y, base_x + text_bbox_size[0], base_y + text_bbox_size[1])

        if use_aliased_text:
            img = img.convert('RGB')

        if np.sum(np.array(img)) == 0:
            print(f"CRITICAL ERROR: Generated text image is blank! Font: {font.path if hasattr(font, 'path') else 'Default'}")

        font_name = 'default'
        if hasattr(font, 'path'):
            if isinstance(font.path, str):
                font_name = os.path.basename(font.path)
            else:
                font_name = 'internal_memory_font'

        metadata = {
            'formatted_text': formatted_text,
            'phi_data': phi_pair['phi_data'],
            'phi_positions': phi_pair['phi_positions'],
            'position': (draw_x, base_y + box_padding),
            'bbox': overall_bbox,
            'font_size': font_size,
            'line_info': line_info,
            'is_single_line': is_single_line,
            'font_used': font_name,
        }

        return img, metadata

    def generate_phi_pair_with_positions(self, seed_val):
        random.seed(seed_val)
        return {
            "formatted_text": f"Patient: TEST {seed_val}",
            "phi_data": {"name": f"TEST {seed_val}"},
            "categories": ["name"],
            "phi_positions": {"name": {"start": 9, "end": 9 + len(f"TEST {seed_val}")}},
            "is_single_line": True,
        }

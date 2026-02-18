
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import io
from typing import Tuple, Dict, Optional, Any
import random


class RealisticPHIDegradation:

    def __init__(self, verbose):
        self.verbose = verbose

    def apply_jpeg_artifacts(self, image, quality):
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, subsampling=0)
        buffer.seek(0)
        compressed = Image.open(buffer).copy()
        buffer.close()
        return compressed

    def apply_sensor_noise(self, image_np, noise_level):

        if noise_level <= 0:
            return image_np
        noisy = image_np.astype(np.float32)
        sigma = noise_level * 25.0
        gaussian_noise = np.random.normal(0, sigma, image_np.shape)
        noisy += gaussian_noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def apply_digital_aliasing(self, image, downsample_factor):

        if downsample_factor >= 1.0:
            return image
        w, h = image.size
        new_w = max(1, int(w * downsample_factor))
        new_h = max(1, int(h * downsample_factor))
        small = image.resize((new_w, new_h), Image.NEAREST)
        pixelated = small.resize((w, h), Image.NEAREST)
        return pixelated

    def texture_text_layer(self, text_np, noise_seed):

        np.random.seed(noise_seed)
        noise_field = np.random.uniform(0.92, 1.0, text_np.shape).astype(np.float32)
        textured_text = text_np.astype(np.float32) * noise_field
        return np.clip(textured_text, 0, 255).astype(np.uint8)

    def degrade_text_region(
        self,xray_image,text_image,text_metadata,degradation_level,seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        bbox = text_metadata['bbox']

        text_region = text_image.crop(bbox)
        xray_region = xray_image.crop(bbox)

        text_np_raw = np.array(text_region.convert('L'))
        xray_np = np.array(xray_region.convert('RGB'))

        text_np = self.texture_text_layer(text_np_raw, noise_seed=seed if seed else 42)

        text_color = (235, 235, 235)

        text_mask = (text_np > 10).astype(np.float32)

        text_colored = np.zeros_like(xray_np)
        for c in range(3):
            text_colored[:, :, c] = (text_np / 255.0) * text_color[c]

        alpha = text_mask[:, :, np.newaxis]
        composite = (text_colored * alpha + xray_np * (1 - alpha)).astype(np.uint8)

        if degradation_level == 'minimal':
            params = {'jpeg_quality': 99, 'noise_level': 0.0, 'downsample': 1.0}
        elif degradation_level == 'light':
            params = {
                'jpeg_quality': random.randint(94, 97),
                'noise_level': random.uniform(0.01, 0.03),
                'downsample': 1.0,
            }
        elif degradation_level == 'moderate':
            params = {
                'jpeg_quality': random.randint(87, 93),
                'noise_level': random.uniform(0.07, 0.15),
                'downsample': random.uniform(0.965, 0.99),
            }
        else:  # heavy
            params = {
                'jpeg_quality': random.randint(75, 85),
                'noise_level': random.uniform(0.15, 0.3),
                'downsample': random.uniform(0.91, 0.95),
            }

        degraded = composite.copy()

        if params['noise_level'] > 0:
            degraded = self.apply_sensor_noise(degraded, noise_level=params['noise_level'])

        if params['downsample'] < 1.0:
            pil_img = Image.fromarray(degraded)
            pil_img = self.apply_digital_aliasing(pil_img, params['downsample'])
            degraded = np.array(pil_img)

        pil_img = Image.fromarray(degraded)
        pil_img = self.apply_jpeg_artifacts(pil_img, params['jpeg_quality'])

        result = xray_image.copy()
        result.paste(pil_img, (bbox[0], bbox[1]))

        metadata = {
            'method': 'textured_digital_overlay',
            'degradation_level': degradation_level,
            'parameters': params,
            'text_color': list(text_color),
        }

        return result, metadata


class RealisticPHIGenerator:
    def __init__(self, verbose):
        self.degrader = RealisticPHIDegradation(verbose=verbose)
        self.verbose = verbose

    def generate_burned_in_phi(
        self,xray_image, text_image,text_metadata,degradation_level,vary_degradation,seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if vary_degradation:
            degradation_level = random.choices(
                ['light', 'moderate', 'heavy'],
                weights=[0.1, 0.35, 0.55],
            )[0]

        return self.degrader.degrade_text_region(
            xray_image, text_image, text_metadata, degradation_level, seed=seed
        )

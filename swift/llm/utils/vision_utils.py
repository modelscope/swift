import base64
import binascii
import math
import os
from io import BytesIO
from typing import Callable, List, TypeVar, Union

import numpy as np
import requests
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def rescale_image(img: 'PIL.Image.Image', rescale_image: int = -1) -> 'PIL.Image.Image':
    import torchvision.transforms as T
    width = img.width
    height = img.height
    if rescale_image <= 0 or width * height <= rescale_image:
        return img

    ratio = width / height
    height_scaled = math.pow(rescale_image / ratio, 0.5)
    width_scaled = height_scaled * ratio
    return T.Resize((int(width_scaled), int(height_scaled)))(img)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size, ((i //
                                                                        (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(img_path: Union[str, 'PIL.Image.Image']) -> 'PIL.Image.Image':
    from PIL import Image, UnidentifiedImageError
    if isinstance(img_path, str):
        img_path = img_path.strip()
        if img_path.startswith('http'):
            content = requests.get(img_path).content
            image = Image.open(BytesIO(content))
        elif os.path.exists(img_path):
            image = Image.open(img_path)
        else:  # base64_str
            try:
                image_data = base64.b64decode(img_path)
                image = Image.open(BytesIO(image_data))
            except (ValueError, binascii.Error, UnidentifiedImageError) as error:
                if len(img_path) < 200:
                    raise ValueError(f'invalid image: "{img_path}"')
                else:
                    raise ValueError(f'invalid image: {error}')
    else:
        image = img_path
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


_T = TypeVar('_T')


def _read_batch(path_list: List[Union[str, 'PIL.Image.Image', None]],
                load_func: Callable[[str], _T] = load_image) -> List[_T]:
    res = []
    for path in path_list:
        if path is None:  # ignore None
            continue
        res.append(load_func(path))
    return res


def transform_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices


def _read_video(video_path: str) -> BytesIO:
    video_path = video_path.strip()
    if video_path.startswith('http'):
        content = requests.get(video_path).content
        mp4_stream = BytesIO(content)
    else:
        with open(video_path, 'rb') as f:
            mp4_stream = BytesIO(f.read())
    return mp4_stream


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    from decord import VideoReader, cpu
    from PIL import Image
    vr = VideoReader(_read_video(video_path), ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def draw_plot(img_dir: str, bbox: List[int], bbox_type: str, output_file: str):
    from PIL import Image, ImageDraw
    from .template import Template
    image = Image.open(img_dir)

    objects = [{'bbox': bbox, 'bbox_type': bbox_type, 'image': 0}]
    Template.normalize_bbox(objects, [image], 'real')
    bbox = objects[0]['bbox']
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline='red', width=2)
    image.save(output_file)


if __name__ == '__main__':
    # A test main to draw bbox
    draw_plot('/mnt/workspace/man.jpg', [354, 462, 580, 738], 'norm_1000', '/mnt/workspace/man_bbox.jpg')

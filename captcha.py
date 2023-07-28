from glob import glob
import random

from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
import numpy as np


class ImageCaptcha:
    """Create an image CAPTCHA.
    captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    width: The width of the CAPTCHA image.
    height: The height of the CAPTCHA image.
    font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width=160, height=60, font_sizes=(42, 50, 56)):
        self._width = width
        self._height = height
        self._font_sizes = font_sizes
        fonts = glob('fonts/*')
        self._truefonts = tuple([
            truetype(n, s)
            for n in fonts
            for s in self._font_sizes
        ])

    @staticmethod
    def create_noise_curve(image, color, count=1, max_thickness=10):
        w, h = image.size
        for _ in range(count):
            color = random_color(0, 200, random.randint(220, 255))
            x1 = random.randint(0, w // 2)
            x2 = random.randint(w // 2 + 1, w)
            y1 = random.randint(0, h - 1)
            y2 = random.randint(y1 + 1, h)
            thickness = random.randint(1, max_thickness)
            points = [x1, y1, x2, y2]
            end = random.randint(0, 359)
            start = random.randint(0, 359)
            Draw(image).arc(points, start, end, fill=color, width=thickness)
        return image

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_image(self, chars, background, return_bbox=False):
        """Create the CAPTCHA image itself.

        chars: text to be generated.
        background: color of the background.

        """
        image = self.generate_background(background[0], background[1])
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self._truefonts)
            w, h = draw.textsize(c, font=font)

            im = Image.new('RGBA', (w, h))
            color = random_color(0, 200, random.randint(220, 255))
            Draw(im).text((0, 0), c, font=font, fill=color)

            # warp
            dx = w * random.uniform(0.0, 0.1)
            dy = h * random.uniform(0.0, 0.1)
            x1 = int(random.uniform(-dx, dx))
            y1 = int(random.uniform(-dy, dy))
            x2 = int(random.uniform(-dx, dx))
            y2 = int(random.uniform(-dy, dy))
            w2 = w + abs(x1) + abs(x2)
            h2 = h + abs(y1) + abs(y2)
            data = (
                x1, y1,
                -x1, h2 - y2,
                w2 + x2, h2 + y2,
                w2 - x2, -y1,
            )
            im = im.resize((w2, h2))
            im = im.transform((w, h), Image.QUAD, data)
            im = im.crop(im.getbbox())
            return im

        images = []
        is_chars = []
        for c in chars:
            is_chars.append(True)
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        offset = int(average * 0.1)

        bboxes = []
        for im, is_char in zip(images, is_chars):
            w, h = im.size
            y = int((self._height - h) / 2)
            y = y + random.randint(-y // 2, y // 2)
            mask = im.getchannel('A')
            x = offset
            image.paste(im, (x, y), mask=mask)
            if is_char:
                bboxes.append([x, y, w, h])
            offset = offset + w + random.randint(-(w // 5), w // 2)

        if width > self._width:
            image = image.resize((self._width, self._height))
            r = self._width / width
            for bbox in bboxes:
                bbox[0] = int(r * bbox[0])
                bbox[2] = int(r * bbox[2])

        if return_bbox:
            return image, bboxes
        else:
            return image

    def generate_image(self, chars, return_bbox=False):
        """Generate the image of the given characters.

        chars: text to be generated.
        """
        background1 = random_color(238, 255)
        background2 = random_color(50, 255)
        background = (background1, background2)
        noise_color = random_color(0, 200, random.randint(220, 255))
        im = self.create_captcha_image(chars, background, return_bbox=return_bbox)
        if return_bbox:
            im, bbox = im
        self.create_noise_dots(im, noise_color, number=60)
        self.create_noise_curve(im, noise_color, 5, 3)
        im = im.filter(ImageFilter.SMOOTH)
        if return_bbox:
            return im, bbox
        else:
            return im

    def generate_background(self, color1, color2):
        if random.random() > 0.5:
            color1, color2 = color2, color1
        
        color = np.linspace(color1, color2, self._width).astype(np.uint8)
        image = np.tile(color, (self._height, 1, 1))
        image = Image.fromarray(image)
        return image


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)

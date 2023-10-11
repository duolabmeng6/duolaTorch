from PIL import Image, ImageDraw, ImageFont
import random
from pyefun import *


def generate_appealing_gradient(width, height):
    image = Image.new('RGB', (width, height))
    pixels = image.load()

    gradient_colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 0), (0, 0, 255), (75, 0, 130),
                       (148, 0, 211)]  # Custom gradient colors
    # 增加10个随机的颜色值到 gradient_colors
    for i in range(10):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        gradient_colors.append((r, g, b))
    # 打乱 gradient_colors
    random.shuffle(gradient_colors)

    for y in range(height):
        for x in range(width):
            # Interpolate between appealing gradient colors based on the x position
            color_index = int(x / width * 6)
            color_fraction = (x / width * 6) - color_index
            r1, g1, b1 = gradient_colors[color_index]
            r2, g2, b2 = gradient_colors[color_index + 1]
            r = int(r1 + (r2 - r1) * color_fraction)
            g = int(g1 + (g2 - g1) * color_fraction)
            b = int(b1 + (b2 - b1) * color_fraction)
            pixels[x, y] = (r, g, b)

    return image


def create_chinese_text_image(text, font_size, image_width, image_height, output_file):
    appealing_gradient_image = generate_appealing_gradient(image_width, image_height)
    draw = ImageDraw.Draw(appealing_gradient_image)
    font = ImageFont.truetype(r'C:\Windows\Fonts\MSYH.TTC', font_size)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    draw.text((4, 4), text, font=font, fill=(r, g, b))  # You can set the text color here
    appealing_gradient_image.save(output_file)


def 文本_取随机汉字(取出的数量: int) -> str:
    '部分常见汉字'
    取出的数量 = 1 if 取出的数量 < 1 else 取出的数量
    文本 = ''
    部分汉字 = "我爱你你知道吗我天天都想着你"
    for x in range(取出的数量):
        文本 += random.choice(部分汉字)
    return 文本


if __name__ == '__main__':
    保存路径 = "./train/"
    保存路径2 = "./test/"
    for i in range(2000):
        chinese_text = 文本_取随机汉字(4)
        create_chinese_text_image(chinese_text, 30, 160, 60, 保存路径 + chinese_text +"_"+ str(i) + '.png')
    # for i in range(100):
    #     chinese_text = 文本_取随机汉字(4)
    #     create_chinese_text_image(chinese_text, 30, 160, 60, 保存路径2 + chinese_text +"_"+ str(i) + '.png')

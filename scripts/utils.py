from PIL import Image
from openpi_client import image_tools
import numpy as np

def crop_left_right(image: Image.Image, left_ratio: float, right_ratio: float) -> Image.Image:
    """
    按比例裁剪左右两边。
    left_ratio: 左边裁掉的比例（0~1）
    right_ratio: 右边裁掉的比例（0~1）
    """
    width, height = image.size
    left = int(width * left_ratio)
    right = int(width * (1 - right_ratio))
    return image.crop((left, 0, right, height))

# 测试代码
if __name__ == "__main__":
    img_path = "/home/ubuntu/pi0/data_300/pick_up_the_cabbage_on_the_plate/0803_181934/image/left_20250803_181943_231061.png"  # 原图路径
    img = Image.open(img_path)

    # 自定义左右裁剪比例（例如左裁 0.2，右裁 0.2）
    cropped_img = crop_left_right(img, left_ratio=0.27, right_ratio=0.13)

    # 再缩放到 224x224
    resized_arr = image_tools.resize_with_pad(np.array(cropped_img), 224, 224)
    resized_img = Image.fromarray(resized_arr)  # 转回 PIL Image 才能保存

    # 保存
    cropped_img.save("cropped.jpg")
    resized_img.save("output_224.jpg")
    # resized_img = cropped_img.resize((224, 224), Image.LANCZOS)

    print(f"原图尺寸: {img.size}")
    print(f"裁剪后尺寸: {cropped_img.size}")
    print("已保存 cropped.jpg（裁剪后） 和 output_224.jpg（缩放到224x224）")

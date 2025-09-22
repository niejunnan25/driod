import pyzed.sl as sl
import cv2
import numpy as np
import os
import copy

class ZedCamera:
    def __init__(self, serial_number=None, device_path=None, resolution=sl.RESOLUTION.HD1080, fps=30):
        """
        初始化 ZED 相机，设置分辨率和帧率。

        参数：
            resolution: ZED 相机分辨率（默认：HD720）
            fps: 帧率（默认：30）
        """

        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = 30

        # 设置设备输入来源（按优先级选择）
        if serial_number is not None:
            input_type = sl.InputType()
            input_type.set_from_serial_number(serial_number)
            init_params.input = input_type
        elif device_path is not None:
            init_params.input = sl.InputType(device_path)

        # 打开相机
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"无法打开 ZED 相机：{err}")
            exit()

        self.image_left = sl.Mat()

        self.frame_count = 0

    def capture_frame(self):
        """
        捕获左右目相机的一帧图像。

        返回：
            tuple: (left_image, right_image) 左右目图像的 NumPy 数组，若捕获失败返回 (None, None)
        """
        right_image = None
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            # 获取左右目图像
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            # self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)

            # 转换为 NumPy 数组, 去掉 alpha 通道
            left_image = self.image_left.get_data()[:,:,:3]
            # right_image = self.image_right.get_data()[:,:,:3]

            self.frame_count += 1

            new_left_image = copy.deepcopy(left_image)

            return new_left_image, right_image
        else:
            print("捕获帧失败")
            return None, None

    def close(self):
        """
        关闭 ZED 相机并释放资源。
        """
        self.zed.close()
        print("ZED Camera has been closed!")

    def __del__(self):

        self.close()

if __name__ == "__main__":
    from PIL import Image

    zed_camera = ZedCamera(serial_number=36276705)
    zed_wrist_camera = ZedCamera(serial_number=13132609)

    left, right = zed_camera.capture_frame()
    left_wrist, right_wrist = zed_wrist_camera.capture_frame()

    left_image_np = np.array(left)
    wrist_image_np = np.array(left_wrist)

    # zed 返回的是 bgr
    Image.fromarray(left_image_np).save("left_image.png")
    Image.fromarray(wrist_image_np).save("wrist_image.png")

    # path = "./test_0"
    # path.mkdir()
    cv2.imwrite("left.png", left)

    cv2.imwrite("left_wrist.png", left_wrist)

    zed_camera.close()
    zed_wrist_camera.close()

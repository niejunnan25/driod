# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
import threading
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
from typing import Optional
import cv2
from utils import crop_left_right

faulthandler.enable()

# DROID data collection frequency -- 控制频率固定为 15Hz
DROID_CONTROL_FREQUENCY = 15
CROP_RATIOS = (0.27, 0.13)

# 全局变量（相机线程会更新）
latest_obs = {}
frames_buffer = {
    "left": [],
    "right": [],
    "wrist": []
}
stop_event = threading.Event()


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"

    # Policy parameters
    external_camera: Optional[str] = (
        "left"  # ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 1200
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "162.105.195.74"
    remote_port: int = 8000

    # Video parameters
    video_fps: int = 30   # 保存视频的帧率


# 防止 Ctrl+C 在关键步骤打断
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def camera_loop(env, args, stop_event):
    """相机线程：独立采集帧"""
    global latest_obs, frames_buffer
    while not stop_event.is_set():
        obs_dict = env.get_observation()
        obs = _extract_observation(args, obs_dict, save_to_disk=False)

        # 更新最新帧
        latest_obs = obs

        # 存视频帧
        if obs["left_image"] is not None:
            frames_buffer["left"].append(obs["left_image"])
        if obs["right_image"] is not None:
            frames_buffer["right"].append(obs["right_image"])
        if obs["wrist_image"] is not None:
            frames_buffer["wrist"].append(obs["wrist_image"])

        # 尽量快，不 sleep 或 sleep 极小
        time.sleep(0.001)


def main(args: Args):
    # 确保外部相机参数正确
    assert args.external_camera in ["left", "right"], f"external_camera 必须是 left 或 right"

    # 初始化环境
    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("Created the droid env!")

    # 启动相机线程
    camera_thread = threading.Thread(target=camera_loop, args=(env, args, stop_event))
    camera_thread.daemon = True
    camera_thread.start()

    # 连接策略服务器
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"成功连接到 {args.remote_host}:{args.remote_port}!")

    # 存储实验结果
    df = pd.DataFrame(columns=["success", "duration", "is_follow", "object", "language_instruction", "video_filename"])

    while True:
        instruction = input("请输入语言指令: ")

        actions_from_chunk_completed = 0
        pred_action_chunk = None

        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        for t_step in bar:
            start_time = time.time()
            try:
                if not latest_obs:
                    continue  # 等待相机第一帧

                curr_obs = latest_obs  # 取最新帧

                # 每 open_loop_horizon 步预测一次
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    request_data = {
                        "observation/image": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/state": np.concatenate(
                            (curr_obs["joint_position"], curr_obs["gripper_position"]), axis=0
                        ),
                        "prompt": instruction,
                    }
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                        pred_action_chunk = pred_action_chunk[:, :8]

                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                if action[-1].item() > 0.20:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                env.step(action)

                # 控制频率锁定 15Hz
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        # 保存视频函数
        def save_video(frames, filename, fps):
            if not frames:
                return
            clip = ImageSequenceClip(frames, fps=fps)
            clip.write_videofile(filename, codec="libx264", audio=False)
            print(f"视频保存完成 {filename}")

        os.makedirs("results/video", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

        save_video(frames_buffer["left"], os.path.join("results/video", f"{timestamp}_third.mp4"), args.video_fps)
        # save_video(frames_buffer["right"], os.path.join("results/video", f"{timestamp}_right.mp4"), args.video_fps)
        save_video(frames_buffer["wrist"], os.path.join("results/video", f"{timestamp}_wrist_left.mp4"), args.video_fps)

        # 成功率输入
        success: str | float | None = None
        while not isinstance(success, float):
            success = input("这次测试成功了吗？(y/n 或输入 0~100 数值)\n")
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0
            else:
                try:
                    success = float(success)
                except:
                    success = None

            if isinstance(success, float) and not (0 <= success <= 1):
                print(f"数值必须在 0~1 之间")
                success = None

        import re

        KNOWN_OBJECTS = ["corn", "green pepper", "red pepper", "garlic", "potato", "cabbage"]

        def extract_object(instruction: str) -> Optional[str]:
            text = instruction.lower()
            for obj in KNOWN_OBJECTS:
                pattern = r"\b" + re.escape(obj) + r"\b"
                if re.search(pattern, text):
                    return obj
            return None

        if success == 1.0:
            is_follow = True
            object = extract_object(instruction)
        else:
            is_follow = False
            object = input("请输入被错误抓取的物体名称: ")

        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        {
                            "success": success,
                            "duration": len(frames_buffer["left"]),
                            "is_follow": is_follow,
                            "object": object,
                            "language_instruction": instruction,
                            "video_filename": f"{timestamp}.mp4",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
                # 清空缓存
        frames_buffer["left"].clear()
        frames_buffer["right"].clear()
        frames_buffer["wrist"].clear()

        if input("再进行一次测试？(y/n) ").lower() != "y":
            break

        env.reset()
        print("环境重设完成...")

    stop_event.set()
    camera_thread.join()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"csv 文件保存完成： {csv_filename}")
    print(df)
    exit()


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False, crop_ratios=CROP_RATIOS):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    def process_image(img):
        if img is None:
            return None
        img = img[..., :3][..., ::-1]  # 去 alpha & 转 RGB
        pil_img = Image.fromarray(img)
        if crop_ratios:
            pil_img = crop_left_right(pil_img, *crop_ratios)
        return np.array(pil_img)

    left_image = process_image(left_image)
    right_image = process_image(right_image)
    wrist_image = process_image(wrist_image)

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        images_to_save = [img for img in [left_image, wrist_image, right_image] if img is not None]
        if images_to_save:
            combined_image = np.concatenate(images_to_save, axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)

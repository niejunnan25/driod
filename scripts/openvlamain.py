# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
import requests
import json_numpy
json_numpy.patch()
from openpi_client import image_tools
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro
from typing import Optional

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 5


@dataclasses.dataclass
class Args:
    left_camera_id: str = "36276705"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "13132609"
    external_camera: Optional[str] = None  # "left" or "right"
    max_timesteps: int = 600
    remote_host: str = "162.105.195.51"
    remote_port: int = 8055


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


def main(args: Args):
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    env = RobotEnv(action_space="cartesian_position", gripper_action_space="position")
    print("Created the droid env!")

    policy_url = f"http://{args.remote_host}:{args.remote_port}/act"

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")

        for t_step in bar:
            start_time = time.time()
            try:
                curr_obs = _extract_observation(
                    args, env.get_observation(), save_to_disk=t_step == 0
                )
                #  图片处理部分
                # image = image_tools.resize_with_pad(
                #     curr_obs[f"{args.external_camera}_image"], 256, 256
                # )
                image = curr_obs[f"{args.external_camera}_image"]
                image = image_transform(image, ifcrop=True)
                video.append(curr_obs[f"{args.external_camera}_image"])

                with prevent_keyboard_interrupt():
                    response = requests.post(
                        policy_url,
                        json={"image": image, "instruction": instruction},
                        timeout=5,
                    )
                if response.status_code != 200:
                    raise RuntimeError(f"Policy server error: {response.text}")

                action = np.array(response.json())
                print(action)
                print("========================================================")
                #current_ee_pose = env.get_ee_pose()
                #print(current_ee_pose)
                #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                #action[:6] = current_ee_pose + action[:6]
                #print(action)
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

                # Binarize gripper action
                if action[-1] > 0.5:
                    action[-1] = 1.0
                else:
                    action[-1] = 0.0

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input("Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100): ")
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0
            else:
                success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be in [0, 100] but got: {success * 100}")

        df = df.append(
            {"success": success, "duration": t_step, "video_filename": save_filename},
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=True):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    if left_image is not None:
        left_image = left_image[..., :3][..., ::-1]
    if right_image is not None:
        right_image = right_image[..., :3][..., ::-1]
    if wrist_image is not None:
        wrist_image = wrist_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        images_to_save = [img for img in [left_image, wrist_image, right_image] if img is not None]
        if images_to_save:
            combined_image = np.concatenate(images_to_save, axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def image_transform(image, ifcrop= False):
    '''
    input:
        image : PIL.Image object or ndarray with shape (1080, 1920)
    output:
        image : PIL.Image object with shape (224, 224), cropped from the left
    '''
    # 如果是 ndarray，先转换为 PIL 图像
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # 原图尺寸
    width, height = image.size  # (1920, 1080)

    # crop 左边的正方形区域（1080 x 1080）
    left = 500
    upper = 0
    right = width  # 保证裁剪为正方形
    lower = height
    if ifcrop == True:
        image = image.crop((left, upper, right, lower))

    # resize 到 (224, 224)
    image = image.resize((224, 224), Image.BILINEAR)

    return np.array(image)


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)


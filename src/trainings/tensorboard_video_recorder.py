import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from subprocess import Popen, PIPE
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, DummyVecEnv

# Disable eager execution for TF v1 compatibility.
tf1.disable_eager_execution()


class TensorboardVideoRecorder(VecEnvWrapper):
    """
    A VecEnv wrapper that records video frames from one of the vectorized environments
    and logs them to TensorBoard as an animated GIF using TensorFlow’s summary API.

    If the provided environment is not vectorized, it will be automatically wrapped in a DummyVecEnv.

    :param env: The environment to wrap (gymnasium.Env or VecEnv).
    :param video_trigger: A function that takes the current global step (int) and returns True
                          when a video should be recorded (e.g., lambda step: step % 10000 == 0).
    :param video_length: The max number of frames to record for the video.
    :param record_video_env_idx: The index of the environment within the vectorized env to record.
    :param tag: Video tag name in TensorBoard.
    :param fps: Frames per second to encode the video.
    :param tb_log_dir: The directory path where TensorBoard logs (summaries) will be saved.
    """

    def __init__(
            self,
            env,
            video_trigger,
            video_length,
            record_video_env_idx=0,
            tag="policy_rollout",
            fps=30,
            tb_log_dir="./logs/tensorboard"
    ):
        # Automatically wrap non-vectorized envs.
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        super().__init__(env)

        self._video_trigger = video_trigger
        self._video_length = video_length
        self._record_video_env_idx = record_video_env_idx
        self._tag = tag
        self._fps = fps

        self._global_step = 0
        self._recording = False
        self._recording_step_count = 0
        self._recorded_frames = []

        self._record_on_reset_pending = False

        self._tb_log_dir = tb_log_dir
        self._file_writer = tf1.summary.FileWriter(tb_log_dir)

    @staticmethod
    def _encode_gif(frames, fps):
        if frames[0].shape[-1] == 4:
            frames = [f[..., :3] for f in frames]

        # Ensure uint8 contiguous frames (ffmpeg rawvideo expects this)
        frames = [np.ascontiguousarray(f.astype(np.uint8)) for f in frames]

        h, w, c = frames[0].shape
        if c not in (1, 3):
            raise ValueError(f"Unexpected channels: {c}, expected 1 or 3")

        pxfmt = {1: "gray", 3: "rgb24"}[c]
        ffmpeg_exe = os.environ.get("IMAGEIO_FFMPEG_EXE", "ffmpeg")

        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-r", f"{fps:.02f}",
            "-s", f"{w}x{h}",
            "-pix_fmt", pxfmt,
            "-i", "-",
            "-vf", "fps=15,scale=320:-1:flags=lanczos",  # <--- reduces load a lot (optional but recommended)
            "-f", "gif",
            "-"
        ]

        proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)

        try:
            for image in frames:
                proc.stdin.write(image.tobytes())
            proc.stdin.close()
            out = proc.stdout.read()
            err = proc.stderr.read()
            ret = proc.wait()
        except BrokenPipeError:
            # ffmpeg died early; pull stderr for the real reason
            err = proc.stderr.read().decode("utf8", errors="replace")
            raise IOError(f"ffmpeg BrokenPipeError.\nCommand: {' '.join(cmd)}\nffmpeg stderr:\n{err}")

        if ret != 0:
            err_txt = err.decode("utf8", errors="replace") if isinstance(err, (bytes, bytearray)) else str(err)
            raise IOError(f"ffmpeg failed (code {ret}).\nCommand: {' '.join(cmd)}\nffmpeg stderr:\n{err_txt}")

        return out

    def _log_video_to_tensorboard(self, tag, video, step):
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            T, H, W, C = video.shape
            summary = tf1.Summary()
            image = tf1.Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = self._encode_gif(list(video), self._fps)
            summary.value.add(tag=tag, image=image)
            self._file_writer.add_summary(summary, step)
        except (IOError, OSError) as e:
            print('GIF summaries require ffmpeg in $PATH.', e)
            tf.summary.image(tag, video, step)

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        if self._recording:
            self._record_frame()
            self._recording_step_count += 1
        return obs

    def _record_frame(self):
        frames = self.venv.env_method("render", indices=[self._record_video_env_idx])
        frame = frames[0]

        # If render() returns None or something unexpected, just skip
        if frame is None:
            return

        frame = np.asarray(frame)

        # Expect (H, W, C) or (H, W)
        if frame.ndim == 2:
            frame = frame[..., None]
        if frame.ndim != 3:
            return

        self._recorded_frames.append(frame)

    def _finalize_video(self):
        if not self._recorded_frames:
            return
        video_np = np.array(self._recorded_frames)  # Shape: (T, H, W, C)
        self._log_video_to_tensorboard(self._tag, video_np, self._global_step)
        self._file_writer.flush()
        self._recording = False
        self._recording_step_count = 0
        self._recorded_frames = []

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._global_step += self.venv.num_envs

        if not self._recording and not self._record_on_reset_pending and self._video_trigger(self._global_step):
            self._record_on_reset_pending = True

        if self._recording:
            self._record_frame()
            self._recording_step_count += 1

            if self._recording_step_count >= self._video_length or dones[self._record_video_env_idx]:
                self._finalize_video()

        if self._record_on_reset_pending and dones[self._record_video_env_idx]:
            self._recording = True
            self._record_on_reset_pending = False
            self._recording_step_count = 0
            self._recorded_frames = []

        return obs, rewards, dones, infos
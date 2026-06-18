from __future__ import annotations

import base64
import os
import tempfile
from dataclasses import dataclass

import cv2
import httpx

from app.core.config import Settings


@dataclass(frozen=True)
class VideoFrame:
    frame_index: int
    source_position: int
    data_url: str


class VideoFrameSampler:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    async def sample(self, video_url: str) -> tuple[VideoFrame, ...]:
        owned_client = self.client is None
        client = self.client or httpx.AsyncClient(
            follow_redirects=True,
            timeout=self.settings.asset_timeout_seconds,
        )
        try:
            response = await client.get(video_url)
            response.raise_for_status()
            if len(response.content) > self.settings.asset_max_download_bytes:
                return ()
            return _sample_video_bytes(response.content, self.settings.max_video_frames)
        except (httpx.HTTPError, OSError, cv2.error):
            return ()
        finally:
            if owned_client:
                await client.aclose()


def _sample_video_bytes(content: bytes, frame_count: int) -> tuple[VideoFrame, ...]:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)

        capture = cv2.VideoCapture(path)
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return ()

            positions = _frame_positions(total_frames, frame_count)
            frames: list[VideoFrame] = []
            for frame_index, position in enumerate(positions):
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                success, frame = capture.read()
                if not success:
                    continue
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    continue
                b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
                frames.append(
                    VideoFrame(
                        frame_index=frame_index,
                        source_position=position,
                        data_url=f"data:image/jpeg;base64,{b64}",
                    )
                )
            return tuple(frames)
        finally:
            capture.release()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _frame_positions(total_frames: int, desired_count: int) -> tuple[int, ...]:
    if desired_count <= 1:
        return (max(0, total_frames // 2),)
    step = max(1, total_frames // desired_count)
    return tuple(min(total_frames - 1, step * index) for index in range(desired_count))

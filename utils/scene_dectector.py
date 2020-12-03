import math

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager, FrameTimecode
from scenedetect.stats_manager import StatsManager

from scenedetect.detectors.content_detector import ContentDetector

def scene_detect(frames, end_time=None):

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    scene_manager.add_detector(ContentDetector())

    base_timecode = FrameTimecode(timecode=0, fps=20)

    start_frame = 0
    curr_frame = 0
    end_frame = None

    total_frames = math.trunc(len(frames))

    start_frame = 0

    scene_manager._start_frame = start_frame

    curr_frame = start_frame

    if start_frame is not None:
        total_frames -= start_frame

    end_frame = len(frames)

    if end_frame is not None:
        total_frames = end_frame

    if total_frames < 0:
        total_frames = 0

    while True:
        if end_frame is not None and curr_frame >= end_frame:
            break

        if (scene_manager._is_processing_required(scene_manager._num_frames + start_frame)
                or scene_manager._is_processing_required(scene_manager._num_frames + start_frame + 1)):
            frame_im = frames[curr_frame]
        else:
            frame_im = None

        scene_manager._process_frame(scene_manager._num_frames + start_frame, frame_im)

        curr_frame += 1
        scene_manager._num_frames += 1

    scene_manager._post_process(curr_frame)

    num_frames = curr_frame - start_frame

    scene_list = scene_manager.get_scene_list(base_timecode)

    frames = []

    for i, scene in enumerate(scene_list):
        frames.append(scene[1].get_frames())

    return frames
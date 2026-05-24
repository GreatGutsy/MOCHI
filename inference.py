# MOCHI inference pipeline 2026
# Feel free to leave suggestions in the issues or discussions tab.

import numpy as np
import cv2
from tqdm import tqdm
import subprocess
import onnxruntime as ort
from config import *

def set_blur(frame, method):
    if method == "HARD_BLUR":
        return cv2.blur(frame, (30, 30))
    elif method == "LIGHT_BLUR":
        return cv2.resize(cv2.resize(frame, (32,18), interpolation=cv2.INTER_LINEAR), (video_shape[0], video_shape[1]), interpolation=cv2.INTER_LINEAR)

def main(INPUT_VIDEO: str, OUTPUT_VIDEO: str, MODEL_PATH: str = "mochi-v1.onnx"):
    opts = ort.SessionOptions()

    model = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(INPUT_VIDEO)
    ret, frame = cap.read()
    video_shape = (frame.shape[1], frame.shape[0]) # W | H
    FPS = round(cap.get(cv2.CAP_PROP_FPS), -1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pre-work variables
    fps_to_10 = FPS // 10
    scale_x = video_shape[0] / 128
    scale_y = video_shape[1] / 128
    t = 0
    avg_delta_luma = 0
    avg_visual_luma = 0
    cell_w_snn = 128 // GRID
    cell_h_snn = 128 // GRID

    command = ffmpeg_command(video_shape, FPS, OUTPUT_VIDEO)

    # if needed
    print(video_shape, FPS) 

    prev_frame = np.ascontiguousarray(np.zeros((3, size[0], size[1]), dtype=np.float32))[None, ...]
    batch = np.empty((2, video_shape[1], video_shape[0], 3))

    proc = subprocess.Popen(command, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)

    mem, spk_rec = np.zeros((1, GRID**2), dtype=np.float32), np.zeros((1, GRID**2), dtype=np.float32)
    thr = np.ones((GRID**2), dtype=np.float32)

    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if t % fps_to_10 == 0:
                current_frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
                img_yuv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YUV)
                
                current_frame = np.ascontiguousarray(img_yuv.transpose(2, 0, 1)).astype(np.float32) * 0.00392157
                current_frame = np.expand_dims(current_frame, axis=0)

                avg_visual_luma += current_frame[0].mean()
                
                # day / night setting
                if avg_visual_luma < 0.13: 
                    target_v = 0.1
                    target_d = 0.005
                else:
                    target_v = 0.28
                    target_d = 0.01175

                delta_frame = np.abs(current_frame - prev_frame)
                avg_delta_luma += delta_frame.mean()
                
                adj_d = np.clip(target_d / (avg_delta_luma / (t + 1)), 0.0005, 0.2)
                adj_v = np.clip(target_v / (avg_visual_luma / (t + 1)), 0.005, 1.0)

                combined_input = np.concatenate([current_frame, delta_frame], axis=1)
                combined_input[:, 0, :, :] *= adj_v
                combined_input[:, 3, :, :] = np.sqrt(combined_input[:, 3, :, :]) * adj_d
                combined_input = np.clip(combined_input, 0.0, 1.0).astype(np.float32)
                combined_input = {'combined_input': combined_input, 'mem': mem, 'spk_rec': spk_rec}

                spk, mem = model.run(['spikes', 'upd_mem'] , combined_input)

                spikes = (spk.flatten() >= thr).astype(bool)

                active_points = []
                for i in range(GRID * GRID):
                    if spikes[i]:
                        row, col = divmod(i, GRID)
                        cx = col * cell_w_snn + cell_w_snn // 2
                        cy = row * cell_h_snn + cell_h_snn // 2
                        active_points.append([cx, cy])

                if len(active_points) > 0:
                    pts = np.array(active_points)

                    if len(pts) >= 3:
                        mean = np.mean(pts, axis=0)
                        std = np.std(pts, axis=0)
                        pts = pts[np.all(np.abs(pts - mean) <= 1.5 * std + 1e-6, axis=1)]
                    
                    if len(pts) > 0:
                        x_min, y_min = np.min(pts, axis=0)
                        x_max, y_max = np.max(pts, axis=0)

                        x1_snn = max(0, x_min - cell_w_snn // 2 - PADDINGTON)
                        y1_snn = max(0, y_min - cell_h_snn // 2 - PADDINGTON // 2)
                        x2_snn = min(128, x_max + cell_w_snn // 2 + PADDINGTON)
                        y2_snn = min(128, y_max + cell_h_snn // 2 + PADDINGTON // 2)

                        x1_hd = int(x1_snn * scale_x)
                        y1_hd = int(y1_snn * scale_y)
                        x2_hd = int(x2_snn * scale_x)
                        y2_hd = int(y2_snn * scale_y)

                        temp_roi = frame[y1_hd:y2_hd, x1_hd:x2_hd].copy()
                        frame = set_blur(frame, BLUR_TYPE)
                        prev_blured = frame
                        frame[y1_hd:y2_hd, x1_hd:x2_hd] = temp_roi
                        
                    else:
                        frame = set_blur(frame, BLUR_TYPE)
                        prev_blured = frame
                else:
                    pts = []
                    frame = set_blur(frame, BLUR_TYPE)
                    prev_blured = frame

                spk_rec = spk
                prev_frame = current_frame
                prev_rect = frame


            else:
                if len(pts) > 0: 
                    temp_roi = frame[y1_hd:y2_hd, x1_hd:x2_hd].copy()
                    frame = prev_blured.copy()
                    frame[y1_hd:y2_hd, x1_hd:x2_hd] = temp_roi
                else:
                    frame = prev_blured.copy()
                    
            if t % fps_to_10 == 1:
                proc.stdin.write(memoryview(prev_rect))
                proc.stdin.write(memoryview(frame))
            elif t % fps_to_10 == 2:
                proc.stdin.write(memoryview(frame))

            t += 1
            if t > 100:
                t = 0
                avg_delta_luma = 0
                avg_visual_luma = 0

            pbar.update(1)

    proc.stdin.close()
    proc.wait()
    print(f"The result is saved into {OUTPUT_VIDEO}")

if __name__ == "__main__":
    INPUT_VIDEO = "video.mp4"
    OUTPUT_VIDEO = "mochi_compressed.mp4"
    MODEL_PATH = "mochi-v1.onnx"
    main(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH)
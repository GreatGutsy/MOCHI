# MOCHI inference pipeline 2026
# Feel free to leave suggestions in the issues or discussions tab.

import csv
import numpy as np
import cv2
from tqdm import tqdm
import subprocess
import onnxruntime as ort
from config import *

# METRICS SETTINGS (PSNR / SSIM / LPIPS)
CALCULATE_METRICS = False      # calculation of ROI metrics
LPIPS_NET         = "alex"
_LPIPS_CACHE = None


def _get_lpips():
    """Lazy LPIPS"""
    global _LPIPS_CACHE
    if _LPIPS_CACHE is None:
        try:
            import torch
            import lpips
            device = "cuda" if torch.cuda.is_available() else "cpu"
            net = lpips.LPIPS(net=LPIPS_NET, verbose=False).to(device).eval()
            _LPIPS_CACHE = (net, device)
        except Exception as e:
            print(f"[metrics] LPIPS недоступен ({e}) -> LPIPS = NaN (pip install lpips)")
            _LPIPS_CACHE = False
    return _LPIPS_CACHE or None


def _bgr_to_lpips_tensor(img_bgr, device):
    """uint8 BGR (H,W,3) -> torch float (1,3,H,W) в диапазоне [-1;1], RGB."""
    import torch
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    t = torch.from_numpy(rgb.astype(np.float32)).div_(127.5).sub_(1.0)
    return t.unsqueeze(0).to(device)


def compute_roi_metrics(roi_before, roi_after, min_side=64):
    nan = float("nan")
    if (roi_before is None or roi_after is None or
            roi_before.size == 0 or roi_after.size == 0 or
            roi_before.shape != roi_after.shape):
        return {"psnr": nan, "ssim": nan, "lpips": nan}

    # ---- PSNR ----
    mse = np.mean((roi_before.astype(np.float64) - roi_after.astype(np.float64)) ** 2)
    psnr_val = float("inf") if mse == 0.0 else 10.0 * np.log10(255.0 ** 2 / mse)

    # ---- SSIM ----
    g1 = cv2.cvtColor(roi_before, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(roi_after,  cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1  = cv2.GaussianBlur(g1 * g1, (11, 11), 1.5) - mu1_sq
    s2  = cv2.GaussianBlur(g2 * g2, (11, 11), 1.5) - mu2_sq
    s12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    ssim_val = float(ssim_map.mean())

    # ---- LPIPS ----
    lpips_val = nan
    pack = _get_lpips()
    if pack is not None:
        net, device = pack
        b, a = roi_before, roi_after
        if min(b.shape[:2]) < min_side:
            k = min_side / min(b.shape[:2])
            w = max(min_side, int(round(b.shape[1] * k)))
            h = max(min_side, int(round(b.shape[0] * k)))
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_CUBIC)
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
        import torch
        with torch.no_grad():
            lpips_val = float(net(_bgr_to_lpips_tensor(b, device),
                                  _bgr_to_lpips_tensor(a, device)).item())

    return {"psnr": psnr_val, "ssim": ssim_val, "lpips": lpips_val}


def log_metrics(history, metrics, frame_idx):
    for k, v in metrics.items():
        history[k].append(v)

def print_metrics_summary(history):
    def mean_of(vals):
        v = np.asarray(vals, dtype=np.float64)
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else float("nan")

    print("\n===== ROI metrics =====")
    print(f"PSNR  mean : {mean_of(history['psnr']):10.2f} dB")
    print(f"SSIM  mean : {mean_of(history['ssim']):10.4f}")
    print(f"LPIPS mean : {mean_of(history['lpips']):10.4f}")

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

    # metrics
    metrics_history = {"psnr": [], "ssim": [], "lpips": []}

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

                        if CALCULATE_METRICS:
                            m = compute_roi_metrics(temp_roi, frame[y1_hd:y2_hd, x1_hd:x2_hd])
                            log_metrics(metrics_history, m, pbar.n + 1)
                            pbar.set_postfix({
                                "PSNR": f"{m['psnr']:.1f}",
                                "SSIM": f"{m['ssim']:.3f}",
                                "LPIPS": f"{m['lpips']:.3f}",
                            })

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
    if CALCULATE_METRICS:
        print_metrics_summary(metrics_history)
    print(f"The result is saved into {OUTPUT_VIDEO}")

if __name__ == "__main__":
    INPUT_VIDEO = "57.mp4"
    OUTPUT_VIDEO = "mochi_57.mp4"
    MODEL_PATH = "mochi-v1.onnx"
    main(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_PATH)

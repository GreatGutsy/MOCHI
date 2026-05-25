# MOCHI: Spiking Neural Networks for Real-Time ROI Selection and Edge-Based Adaptive Compression

MOCHI (Memory Optimized Compression Hybrid Interface) is an end-to-end framework that unifies high-speed Spiking Neural Networks (SNNs) and intelligent video compression for low-power edge hardware. Optimized for deployment on commodity single-board computers without dedicated GPU/NPU accelerators, the system eliminates the standard CPU bottleneck caused by software-bound video encoding.

### Key Features
* **High-Speed ROI Isolation**: Implements an optimized single-step SNN core with inter-frame recursion, achieving a 6.6× inference speedup over a YOLO26n baseline pipeline.
* **Spatial Noise Suppression**: Introduces a cellular-automata suppression mechanism based on a 1.5σ threshold rule to prevent false-positive spikes.
* **Storage Optimization**: Drives extreme FFmpeg bitrate reduction by aggressively blurring static backgrounds while fully preserving ROI feature quality, extending the video archive depth by 20% to 30%.

### Tested with
```text
Python == 3.13.5
NumPy == 2.4.6
onnxruntime == 1.26.0
opencv_python == 4.12.0.88
tqdm == 4.67.1
```
### Examples
*Link to Hugging Face will be here soon*

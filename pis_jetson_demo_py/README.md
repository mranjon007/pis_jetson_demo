PIS_Jetson_Py
=============
PIS 과제 3차년도 Python 데모 레포지토리 (모델 변환 후 데모까지의 전체 파이프라인 포함)

## Contents
1. [Requirements](#requirements)  
  1-1. [PC (Development environment)](#pc-development-environment)  
  1-2. [Jetson (Production environment)](#jetson-production-environment)  
2. [Model conversion](#model-conversion)
3. [Inference pipeline](#inference-pipeline)

-----

## Requirements
### PC (Development environment)
```bash
# docker images are in 172.30.1.60 server
docker run -it --gpus all --name pis-tensorrt jian/pis-tensorrt:8205
```
Base environment: `Docker` with image `nvcr.io/nvidia/tensorrt:22.05-py3`  
See details on [TensorRT Container Release Notes Documentation](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/index.html#rel_22-05)
- OpenCV 4.x: `sudo apt-get install libopencv4-dev`
- (Optional) GStreamer 1.0 plugins: `sudo apt-get install gstreamer1.0-plugins-ugly gstreamer1.0-rtsp`
  - Required when using GStreamer as an output(live preview)


### Jetson (Production environment)
Base environment: Native OS (Jetson Jetpack 5.1.1)  
See details on [Jetson Jetpack SDK 5.1.1 - Key Features in Jetpack](https://developer.nvidia.com/embedded/jetpack-sdk-511#jetpack-features)
- OpenCV 4.x: Preinstalled (or install manually: `sudo apt-get install libopencv4-dev`)
- TensorRT 8.5.0.2, CUDA 11.4: Preinstalled (or install manually using [sdkmanager](https://developer.nvidia.com/sdk-manager))
- (Optional) GStreamer 1.0 plugins: `sudo apt-get install gstreamer1.0-plugins-ugly gstreamer1.0-rtsp`
  - Required when using GStreamer as an output(live preview)

## Model conversion
This repository contains easy-to-use toolkit for converting ONNX models into TensorRT engine (plan) file.
- ONNX model, `.env` file storage: `\\172.30.1.5\연구소\개인폴더\안정인\2023-08-PIS\onnx-weights`

### Conversion: Usage
Use `./models/build.sh` to convert ONNX model to TensorRT engine plan file. Below describes the detailed instructions.

1. Put your ONNX model into `./models/onnx/` directory, and rename with following template:  
    ```bash
    <model_name>.<type>.onnx
    # e.g. Yolo-NAS-S.with-parser.onnx
    ```
2. Copy `./models/onnx/.template.env` into new file and rename with same basename of the onnx filename. (For instance, save as `./models/onnx/Yolo-NAS-S.with-parser.env`)

3. Change directory to `./model/` and run `./build.sh`

## Inference pipeline
Make sure to install requirements with `requirements.txt` before running any demo.
```bash
python3 -m pip install requirements.txt
```

### Inference: Getting Started
<details>
  <summary>by Source (Camera, File, RTSP URL, etc)</summary>

```bash
# Local USB camera source (device ID 0), File destination
# You should Ctrl+C to interrupt streaming (Objects will be normally destructed)
python3 demo.py --config-name sample.pms3.camsource.filesink
```
```bash
# File source, File destination
python3 demo.py --config-name sample.pms3.filesource.filesink
```
```bash
# RTSP source, File destination
# You should Ctrl+C to interrupt streaming (Objects will be normally destructed)
python3 demo.py --config-name sample.pms3.livesource.filesink
```
```bash
# Video folder (bulk inference)
python3 demo.py --config-name sample.pms3.videofolder
```
</details>


<details>
  <summary>by Destination (Camera, File, RTSP URL, etc)</summary>

```bash
# Physical display output
python3 demo.py --config-name sample.pms3.filesource.x11sink
```
```bash
# File output
python3 demo.py --config-name sample.pms3.filesource.filesink
```
```bash
# Launch from RTSP source to RTSP destination (RTSP server live preview)
# Requires OpenCV gstreamer support + gstreamer rtsp plugins
python3 demo.py --config-name sample.pms3.livesource.gstsink
```
```bash
# Video folder (bulk inference)
python3 demo.py --config-name sample.pms3.videofolder
```
</details>


### Inference: Usage
```bash
# Copy file from recipes/sample.pms3.filesource.filesink.yaml
# to recipes/your-recipe-name.yaml

python3 demo.py --config-name your-recipe-name
```
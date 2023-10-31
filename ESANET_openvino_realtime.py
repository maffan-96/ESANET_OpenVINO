# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""

import cv2
import numpy as np
import open3d
from open3d.visualization import Visualizer
import pyrealsense2 as rs
import openvino.runtime as ov
from openvino.runtime import AsyncInferQueue, InferRequest

global time_m
time_m = 0

global color
color = [
    [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
    [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
    [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
    [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
    [128, 64, 128], [0, 192, 128], [128, 192, 128], [64, 64, 0],
    [192, 64, 0], [64, 192, 0], [192, 192, 0], [64, 64, 128],
    [192, 64, 128], [64, 192, 128], [192, 192, 128], [0, 0, 64],
    [128, 0, 64], [0, 128, 64], [128, 128, 64], [0, 0, 192],
    [128, 0, 192], [0, 128, 192], [128, 128, 192], [64, 0, 64]
]

vis = Visualizer()
vis.create_window('PCD', width=640, height=480)
pointcloud = open3d.geometry.PointCloud()
geometry_added = False

pred_colored = None

def postprocess(prediction, color):
    global pred_colored
    pred = np.argmax(prediction, axis=1)
    pred = pred.squeeze().astype(np.uint8)
    cmap = np.asarray(color, dtype='uint8')
    pred_colored = cmap[pred]

def callback(infer_request: InferRequest, userData: None) -> None:
    global color
    pred_ir = next(iter(infer_request.results.values()))
    postprocess(pred_ir, color)

# Setup RealSense pipeline
config = rs.config()
pipeline = rs.pipeline()

# Enable pipeline
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device_realsense = pipeline_profile.get_device()
device_product_line = str(device_realsense.get_info(rs.camera_info.product_line))

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start RealSense pipeline
pipeline.start(config)
align_to_color = rs.align(rs.stream.color)

dstream = pipeline_profile.get_stream(rs.stream.depth)
dstream_prof = dstream.as_video_stream_profile()
print("**************+", dstream_prof.get_intrinsics())
intr_1 = dstream_prof.get_intrinsics()

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    frames = pipeline.wait_for_frames()
    frames = align_to_color.process(frames)

def itr_time(start_t, stop_t, i, infr_mod):
    duration = stop_t - start_t
    avg_duration = duration / i
    print("Avg Loop time = ", avg_duration, flush=True)
    print("Avg Model Inf time = ", infr_mod / i)

import time

# Initialize OpenVINO core
core = ov.Core()
print("Available devices: ", core.available_devices)
device = "GPU"
model_path = "/home/maffan/affan/ESANet-affan/onnx_models/FP32/model.xml"
model = core.read_model(model=model_path)

def my_preprocess(samp, IMAGE_WIDTH=640, IMAGE_HEIGHT=480):
    IMAGE, DEPTH = samp['image'], samp['depth']
    IMAGE = cv2.resize(IMAGE, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    DEPTH = cv2.resize(DEPTH, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    input_image = IMAGE
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_image = input_image / 255
    input_image = (input_image - mean) / std
    input_image = np.transpose(input_image, axes=(2, 0, 1))
    depth_mean = 2841.94941272766
    depth_std = 1417.2594281672277
    input_depth = np.expand_dims(DEPTH, 0).astype('float32')
    depth_0 = input_depth == 0
    input_depth = (input_depth - depth_mean) / depth_std
    input_depth[depth_0] = 0
    samp['image'] = input_image
    samp['depth'] = input_depth
    return samp

compiled_model = core.compile_model(model, device, config={"PERFORMANCE_HINT": "LATENCY"})
input_layer1 = compiled_model.input(0)
input_layer2 = compiled_model.input(1)
output_layer = compiled_model.output(0)

print("Input RGB image shape from OpenVINO model = ", input_layer1.shape, input_layer2.shape)

if __name__ == '__main__':
    device = "CPU"

    # Inference
    start_time = time.time()
    itr = 1

    # Asynchronous computation starts here
    infer_queue = AsyncInferQueue(compiled_model, 2)
    infer_queue.set_callback(callback)

    while True:
        next_frames = pipeline.wait_for_frames()
        next_frames = align_to_color.process(next_frames)
        next_depth_frame = next_frames.get_depth_frame()
        next_color_frame = next_frames.get_color_frame()
        next_depth_image = np.asanyarray(next_depth_frame.get_data())
        next_color_image = np.asanyarray(next_color_frame.get_data())

        next_sample2 = my_preprocess({'image': next_color_image, 'depth': next_depth_image})
        next_image = np.ascontiguousarray(next_sample2['image'][None], np.float32)
        next_depth2 = np.ascontiguousarray(next_sample2['depth'][None])

        start_m = time.perf_counter()
        next_input_tensor1 = ov.Tensor(next_image, shared_memory=True)
        next_input_depth2 = ov.Tensor(next_depth2)

        infer_queue.start_async({input_layer1: next_input_tensor1, input_layer2: next_input_depth2})
        end_time_model = time.perf_counter()
        time_m = time_m + end_time_model - start_m
        itr = itr + 1

        if pred_colored is not None:
            seg_image = pred_colored
            img3 = cv2.hconcat((next_color_image, seg_image))

            depth_image = np.asarray(next_depth_image)

            color_raw = open3d.geometry.Image(seg_image)
            depth_raw = open3d.geometry.Image(next_depth_image)

            rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=4000.0, convert_rgb_to_intensity=False)
            pinhole_camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(intr_1.width, intr_1.height, intr_1.fx, intr_1.fy, intr_1.ppx, intr_1.ppy)

            pcd_1 = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
            pcd_1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pcd_1 = pcd_1.voxel_down_sample(voxel_size=0.001)
            pointcloud.points = pcd_1.points
            pointcloud.colors = pcd_1.colors

            if not geometry_added:
                vis.add_geometry(pointcloud)
                geometry_added = True

            vis.update_geometry(pointcloud)
            vis.poll_events()
            vis.update_renderer()

            if True:
                cv2.imshow("Semantic Segmentation", img3)
                cv2.imshow('Depth Stream', depth_image)
                if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                    stop_time = time.time()
                    itr_time(start_time, stop_time, itr, time_m)
                    vis.destroy_window()
                    raise StopIteration

        infer_queue.wait_all()


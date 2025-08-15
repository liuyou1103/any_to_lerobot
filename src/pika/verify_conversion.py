# import os
# import numpy as np
# import imageio.v3 as imageio
# import cv2
# from pathlib import Path

# def load_raw_depth_png(png_path):
#     """加载原始PNG深度图像并转换为三维单通道数组"""
#     png_path = Path(png_path)
#     if not png_path.exists():
#         raise FileNotFoundError(f"原始PNG文件不存在：{png_path}")
    
#     # 读取PNG图像（支持16位/8位单通道）
#     raw_img = imageio.imread(png_path)
    
#     # 确保格式为 (H, W, 1)（三维单通道）
#     if raw_img.ndim == 2:
#         raw_img = np.expand_dims(raw_img, axis=-1)
#     return raw_img

# def extract_depth_frame_from_video(video_path, frame_index=0):
#     """从FFV1编码的深度视频中提取指定帧并转换为三维单通道数组"""
#     video_path = Path(video_path)
#     if not video_path.exists():
#         raise FileNotFoundError(f"视频文件不存在：{video_path}")
    
#     # 打开视频文件
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise IOError(f"无法打开视频文件：{video_path}")
    
#     # 跳转到目标帧
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
#     ret, frame = cap.read()
#     cap.release()
    
#     if not ret:
#         raise ValueError(f"无法读取视频中的第{frame_index}帧")
    
#     # 转换为单通道（深度视频为灰度图，需去除冗余通道）
#     # 注：OpenCV读取单通道视频时可能返回3通道（BGR相同），需转为灰度
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # 转为 (H, W, 1) 格式
#     video_img = np.expand_dims(gray_frame, axis=-1)
#     return video_img

# def verify_depth_consistency(raw_png_path, video_path, frame_index=0):
#     """验证原始PNG与视频帧的深度信息是否一致"""
#     # 加载原始PNG和视频帧
#     raw_img = load_raw_depth_png(raw_png_path)
#     video_img = extract_depth_frame_from_video(video_path, frame_index)
    
#     # 1. 检查形状是否一致
#     if raw_img.shape != video_img.shape:
#         print(f"❌ 形状不一致：原始PNG {raw_img.shape}，视频帧 {video_img.shape}")
#         return False
    
#     # 2. 检查数据类型是否一致（如uint16/uint8）
#     if raw_img.dtype != video_img.dtype:
#         print(f"❌ 数据类型不一致：原始PNG {raw_img.dtype}，视频帧 {video_img.dtype}")
#         return False
    
#     # 3. 逐像素比对数值
#     diff = np.abs(raw_img - video_img)
#     max_diff = diff.max()
#     total_diff_pixels = np.sum(diff > 0)
    
#     if max_diff == 0 and total_diff_pixels == 0:
#         print(f"✅ 深度信息完全一致（形状：{raw_img.shape}，数据类型：{raw_img.dtype}）")
#         return True
#     else:
#         print(f"❌ 存在差异：")
#         print(f"  - 最大像素差值：{max_diff}")
#         print(f"  - 差异像素总数：{total_diff_pixels}")
#         print(f"  - 差异像素占比：{total_diff_pixels / raw_img.size:.2%}")
#         return False

# if __name__ == "__main__":
#     # --------------------------
#     # 配置参数（需根据实际文件路径修改）
#     # --------------------------
#     RAW_PNG_PATH = "/home/ctos/pika/nas/task_bag_todesk_1000_4.29/1/episode1/camera/depth/pikaDepthCamera_c/1745856310.222979.png"  # 原始PNG深度图像路径
#     VIDEO_PATH = "/home/ctos/pika/convert/task_bag_todesk_1000_4.29/videos/chunk-000/observation.depths.cam_center/episode_000000.avi"  # 编码后的深度视频路径
#     TARGET_FRAME_INDEX = 1  # 要验证的帧索引（需与原始PNG对应）
    
#     # 执行验证
#     verify_depth_consistency(RAW_PNG_PATH, VIDEO_PATH, TARGET_FRAME_INDEX)
# import subprocess
# import numpy as np
# import cv2
# from pathlib import Path

# def save_first_frame_as_png(video_path: Path | str, output_png_path: Path | str) -> None:
#     """Save the first frame of a video as a lossless 16-bit PNG."""
#     video_path = Path(video_path)
#     output_png_path = Path(output_png_path)
#     output_png_path.parent.mkdir(parents=True, exist_ok=True)

#     # 使用 FFmpeg 解码第一帧并保存为 16-bit PNG
#     ffmpeg_cmd = [
#         "ffmpeg",
#         "-i", str(video_path),
#         "-frames:v", "1",          # 只解码一帧
#         "-pix_fmt", "gray16le",    # 保持 16-bit 精度
#         "-color_range", "1",       # full range (0-65535)
#         "-y",                      # 覆盖输出文件
#         str(output_png_path)
#     ]
#     subprocess.run(ffmpeg_cmd, check=True)

#     # 验证输出图片的精度
#     img = cv2.imread(str(output_png_path), cv2.IMREAD_UNCHANGED)
#     if img is None:
#         raise ValueError("Failed to read the saved PNG.")
#     if img.dtype != np.uint16:
#         raise ValueError(f"Output PNG is not uint16 (got {img.dtype}).")

#     print(f"Saved first frame (uint16) to {output_png_path}")

# # 示例调用
# save_first_frame_as_png("/home/ctos/pika/convert/task_bag_todesk_1000_4.29/videos/chunk-000/observation.depths.cam_center/episode_000000.avi", "first_frame_16bit.png")

# import cv2
# import numpy as np

# def are_images_identical(img1_path, img2_path):
#     # 读取图片（保留原始精度，如 uint16）
#     img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
    
#     if img1 is None or img2 is None:
#         raise ValueError("Failed to read one or both images.")
#     print(img1.shape,img2.shape)
#     # 检查尺寸和通道数是否一致
#     if img1.shape != img2.shape:
#         return False
    
#     # 直接比较像素值（支持 uint8/uint16 等）
#     return np.array_equal(img1, img2)

# # 示例调用
# result = are_images_identical("/home/ctos/frame_000000.png", "/home/ctos/pika/nas/task_bag_todesk_1000_4.29/2/episode1/camera/depth/pikaDepthCamera_c/1745856339.949802.png")
# print("Images are identical:", result)






# import cv2
# import numpy as np

# def detect_avi_pixel_type(video_path):
#     """
#     检测AVI视频的像素数据类型（uint8或uint16）
    
#     参数:
#         video_path: AVI视频文件的路径
        
#     返回:
#         str: 像素数据类型（'uint8'、'uint16'或'unknown'）
#     """
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         raise ValueError(f"无法打开视频文件: {video_path}")
    
#     try:
#         # 读取第一帧
#         ret, frame = cap.read()
        
#         if not ret:
#             raise ValueError("无法读取视频帧")
        
#         # 获取帧的数据类型
#         dtype = frame.dtype
        
#         # 检查数据类型
#         if dtype == np.uint8:
#             return 'uint8'
#         elif dtype == np.uint16:
#             return 'uint16'
#         else:
#             return f'unknown (实际类型: {dtype})'
    
#     finally:
#         # 释放资源
#         cap.release()

# if __name__ == "__main__":
#     # 在这里直接写死视频路径
#     video_path = "/home/ctos/pika/convert/task_bag_todesk_1000_4.29/videos/chunk-000/observation.depths.cam_center/episode_000000.avi"  # Windows系统示例
#     # video_path = "/home/user/videos/your_video.avi"  # Linux/macOS系统示例
#     # video_path = "relative/path/to/video.avi"  # 相对路径示例（与脚本同目录时直接写文件名）
    
#     try:
#         pixel_type = detect_avi_pixel_type(video_path)
#         print(f"视频像素数据类型: {pixel_type}")
#     except Exception as e:
#         print(f"错误: {str(e)}")


# import subprocess
# from pathlib import Path

# def get_video_pixel_format(video_path):
#     video_path = Path(video_path)
#     if not video_path.exists():
#         raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
#     ffprobe_args = [
#         "ffprobe",
#         "-v", "error",
#         "-select_streams", "v:0",
#         "-show_entries", "stream=pix_fmt",
#         "-of", "default=noprint_wrappers=1:nokey=1",
#         str(video_path)
#     ]
    
#     try:
#         result = subprocess.run(
#             ffprobe_args,
#             check=True,
#             capture_output=True,
#             text=True
#         )
#         pix_fmt = result.stdout.strip()
#         if not pix_fmt:
#             raise ValueError("无法检测视频的像素格式")
#         return pix_fmt
#     except subprocess.CalledProcessError as e:
#         # 显示ffprobe的错误信息
#         raise RuntimeError(f"检测像素格式失败: {e.stderr}")

# def save_first_frame_as_png(video_path, output_png_path, overwrite=False):
#     video_path = Path(video_path)
#     output_png_path = Path(output_png_path)
    
#     # 确保输出目录存在
#     if output_png_path.suffix == "":  # 如果输出路径是目录，自动生成文件名
#         output_png_path = output_png_path / "first_frame.png"
#     output_png_path.parent.mkdir(parents=True, exist_ok=True)
    
#     pix_fmt = get_video_pixel_format(video_path)
#     print(f"检测到视频像素格式: {pix_fmt}")
    
#     ffmpeg_args = [
#         "ffmpeg",
#         "-i", str(video_path),
#         "-vframes", "1",
#         "-pix_fmt", pix_fmt,
#         "-compression_level", "0",
#     ]
    
#     if overwrite:
#         ffmpeg_args.append("-y")
    
#     ffmpeg_args.append(str(output_png_path))
    
#     try:
#         # 移除 stdout 和 stderr 的重定向，显示完整输出
#         result = subprocess.run(
#             ffmpeg_args,
#             check=True,
#             capture_output=True,
#             text=True
#         )
#         print(f"已按 {pix_fmt} 格式保存第一帧到: {output_png_path}")
#     except subprocess.CalledProcessError as e:
#         # 显示ffmpeg的详细错误
#         raise RuntimeError(
#             f"提取帧失败: {e}\n"
#             f"stdout: {e.stdout}\n"
#             f"stderr: {e.stderr}"
#         )

# if __name__ == "__main__":
#     video_path = "/home/ctos/pika/convert/task_fold_pants_1000_4.31/videos/chunk-000/observation.depths.cam_center/episode_000000.avi"
#     output_path = "/home/ctos/first_frame.png"  # 明确指定文件名，而非目录
#     save_first_frame_as_png(video_path, output_path, overwrite=True)





# import cv2
# import numpy as np

# img = cv2.imread("/home/ctos/first_frame.png", cv2.IMREAD_UNCHANGED)
# print(f"数据类型: {img.dtype}")  # 应输出 uint16
# print(f"像素值范围: {img.min()} ~ {img.max()}")  # 16位数据范围通常是 0~65535





# import cv2
# import numpy as np
# from pathlib import Path

# def are_images_identical(img_path1: Path | str, img_path2: Path | str) -> bool:
#     """
#     检查两张图片是否完全一致（形状、数据类型、所有像素值均相同）
    
#     参数:
#         img_path1: 第一张图片的路径
#         img_path2: 第二张图片的路径
        
#     返回:
#         若完全一致则返回True，否则返回False
#     """
#     # 转换为Path对象
#     img_path1 = Path(img_path1)
#     img_path2 = Path(img_path2)
    
#     # 检查文件是否存在
#     if not img_path1.exists():
#         raise FileNotFoundError(f"图片文件不存在: {img_path1}")
#     if not img_path2.exists():
#         raise FileNotFoundError(f"图片文件不存在: {img_path2}")
    
#     # 读取图片（保留原始格式和精度）
#     img1 = cv2.imread(str(img_path1), cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread(str(img_path2), cv2.IMREAD_UNCHANGED)
    
#     # 检查读取是否成功
#     if img1 is None:
#         raise ValueError(f"无法读取图片: {img_path1}")
#     if img2 is None:
#         raise ValueError(f"无法读取图片: {img_path2}")
    
#     # 检查形状是否相同（包括高度、宽度和通道数）
#     if img1.shape != img2.shape:
#         print(f"形状不同: {img1.shape} vs {img2.shape}")
#         return False
    
#     # 检查数据类型是否相同
#     if img1.dtype != img2.dtype:
#         print(f"数据类型不同: {img1.dtype} vs {img2.dtype}")
#         return False
    
#     # 检查所有像素值是否相同
#     if not np.array_equal(img1, img2):
#         # 找出不同的像素位置（可选）
#         diff_mask = img1 != img2
#         diff_count = np.count_nonzero(diff_mask)
#         print(f"存在 {diff_count} 个不同的像素")
#         return False
    
#     # 所有检查通过
#     print("两张图片完全一致")
#     return True

# # 使用示例
# if __name__ == "__main__":
#     # 替换为你要对比的两张图片路径
#     image1_path = "/home/ctos/first_frame.png"
#     image2_path = "/home/ctos/pika/nas/task_bag_todesk_1000_4.29/1/episode1/camera/depth/pikaDepthCamera_c/1745856310.289195.png"
    
#     try:
#         result = are_images_identical(image1_path, image2_path)
#         print(f"对比结果: {'完全一致' if result else '存在差异'}")
#     except Exception as e:
#         print(f"对比失败: {e}")
# import numpy as np
# import imageio.v3 as imageio
# def load_image(image_path):
#     if isinstance(image_path, np.ndarray):
#         img = image_path
#     else:
#         img = imageio.imread(image_path)
#     # 如果是灰度图像 (H, W)，扩展成 (H, W, 1)
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=-1)  # 在最后增加一个维度
#     print(f"Loaded image dtype: {img.dtype}, shape: {img.shape}")
#     return img

# original_path = '/home/ctos/pika/convert/task_fold_pants_1000_4.31/images/observation.depths.cam_center/episode_000000/frame_000000.png'
# saved_path = '/home/ctos/pika/nas/task_fold_pants_1000_4.31/1/episode1/camera/depth/pikaDepthCamera_c/1745803414.063253.png'
# original_array = load_image(original_path)
# saved_array = load_image(saved_path)
# print(np.array_equal(original_array, saved_array))  # 应返回True
import numpy as np
import imageio.v3 as imageio
 
def load_image(image_path):
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = imageio.imread(image_path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    print(f"Loaded image dtype: {img.dtype}, shape: {img.shape}")
    return img
 
# 原始路径和保存路径
saved_path = '/home/ctos/first_frame.png'
original_path = '/home/ctos/pika/nas/task_fold_pants_1000_4.31/1/episode1/camera/depth/pikaDepthCamera_c/1745803414.096562.png'
 
# 加载原始图像
original_array = load_image(original_path)
 
# 方法1：用imageio重新保存并加载
temp_path = "temp_debug.png"
imageio.imwrite(temp_path, original_array.squeeze())  # 保存为2D
temp_array = load_image(temp_path)
print("Symmetry test (imageio):", np.array_equal(original_array, temp_array))
 
# 方法2：直接比较原始和目标图像
saved_array = load_image(saved_path)
print("Original vs Saved:", np.array_equal(original_array, saved_array))
 
# 检查差异
diff = np.abs(original_array.astype(np.int32) - saved_array.astype(np.int32))
print("Max difference:", diff.max())
print("Number of different pixels:", np.sum(diff > 0))

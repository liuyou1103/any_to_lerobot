'''
version v1.0
'''

import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Callable
import subprocess
from collections import defaultdict, OrderedDict


class DoRobotDataMerger:
    def __init__(self, repo_url: Optional[str] = None, data_dirs: Union[str, List[str]] = None, output_dir: str = None, log_to_file: bool = False):
        """
        初始化数据合并工具

        Args:
            repo_url: 代码仓库URL，用于自动更新代码版本
            data_dirs: 单个数据目录路径(字符串)或多个目录路径(列表)
            output_dir: 合并输出目录
            log_to_file: 是否将日志保存到文件（默认False，输出到控制台）
        """
        self.repo_url = repo_url
        self.data_dirs = self._normalize_data_dirs(data_dirs) if data_dirs else []
        self.output_dir = Path(output_dir)
        # 设置日志
        self._setup_logging(log_to_file)
        logging.info(f"初始化数据目录: {self.data_dirs}")
        self.current_code_version = "v1.0"
        self.supported_data_versions = ["v1.0"]
        self.existing_dirs_list = []

    def _setup_logging(self, log_to_file: bool):
        """配置日志输出（始终输出到终端，仅在 log_to_file=True 时输出到文件）"""
        # 清除现有 handlers，避免重复日志
        logging.getLogger().handlers.clear()
    
        # 设置日志格式和级别
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[logging.StreamHandler()]  # 始终输出到终端
        )
    
        # 如果需要，额外添加文件日志
        if log_to_file:
            file_handler = logging.FileHandler("data_merger.log")
            file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
            logging.getLogger().addHandler(file_handler)

    def _normalize_data_dirs(self, data_dirs: Union[str, List[str]]) -> List[Path]:
        """
        标准化数据目录输入：
        - 如果是字符串路径，获取其所有子目录
        - 如果是列表，确保所有元素都是Path对象

        Args:
            data_dirs: 字符串路径或路径列表

        Returns:
            Path对象的列表

        Raises:
            FileNotFoundError: 路径不存在
            ValueError: 输入类型无效
        """
        if isinstance(data_dirs, str):
            path = Path(data_dirs)
            if not path.exists():
                raise FileNotFoundError(f"路径不存在: {path}")
            if path.is_file():
                return [path.parent]
            else:
                subdirs = [d for d in path.iterdir() if d.is_dir()]
                
                def extract_sort_key(dirname):
                    parts = str(dirname).split('_')
                    if parts:
                        last_part = parts[-1]
                        if last_part.isdigit():
                            return int(last_part)
                    return 0
                
                subdirs.sort(key=extract_sort_key)
                return subdirs
        
        elif isinstance(data_dirs, list):
            normalized = []
            for d in data_dirs:
                p = Path(d) if isinstance(d, str) else d
                if not p.exists():
                    raise FileNotFoundError(f"路径不存在: {p}")
                normalized.append(p)
            return normalized
        
        else:
            raise ValueError("data_dirs 必须是字符串或字符串列表")

    @staticmethod
    def get_img_video_path(each_task_path: Path, camera_images: str, camera_images_path: Path) -> tuple[List[Path], List[Path], List[Path]]:
        """
        获取图像和视频路径列表

        Args:
            each_task_path: 任务根目录路径
            camera_images: 相机名称
            camera_images_path: 相机图像目录路径

        Returns:
            包含三个列表的元组：(img_path_list, video_path_list, label_video_path_list)
        """
        img_path_list = []
        video_path_list = []
        label_video_path_list = []
        
        try:
            for episode_index in camera_images_path.iterdir():
                if not episode_index.is_dir():
                    continue
                    
                episode_index_name = episode_index.name
                video_name = episode_index_name + '.avi'
                label_video_name = episode_index_name + '.mp4'
                video_path = each_task_path / "videos" / "chunk-000" / camera_images / video_name
                label_video_path = each_task_path / "label" / "chunk-000" / camera_images / label_video_name
                
                # 确保目录存在
                video_path.parent.mkdir(parents=True, exist_ok=True)
                label_video_path.parent.mkdir(parents=True, exist_ok=True)
                
                img_path_list.append(episode_index)
                video_path_list.append(video_path)
                label_video_path_list.append(label_video_path)

            if len(img_path_list) != len(video_path_list):
                logging.error("图像和视频路径数量不匹配")
                return [], [], []
                
            logging.debug(f"找到 {len(img_path_list)} 组图像和视频路径")
            return img_path_list, video_path_list, label_video_path_list
        except Exception as e:
            logging.error(f"获取路径列表失败: {str(e)}")
            return [], [], []

    @staticmethod
    def read_info_json(each_task_path: Path) -> Optional[int]:
        """
        读取info.json文件获取FPS值

        Args:
            each_task_path: 包含meta/info.json的目录路径

        Returns:
            FPS值，读取失败返回None
        """
        try:
            with open(each_task_path / "info.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                fps = data.get("fps")
                logging.debug(f"读取到FPS值: {fps}")
                return fps
        except Exception as e:
            logging.error(f"读取info.json失败: {str(e)}")
            return None

    def _compile_images_to_videos(self, input_dirs: List[Path]) -> None:
        """
        将图像序列编译为视频文件（.avi和.mp4格式）

        Args:
            input_dirs: 包含images目录的输入路径列表
        """
        for input_dir in input_dirs:
            images_dir = input_dir / "images"
            meta_dir = input_dir / "meta"
            fps = self.read_info_json(meta_dir)
            if fps is None:
                raise ValueError("无法读取FPS值，跳过编译")

            camera_images_path_name = []
            for camera_images_path in images_dir.iterdir():
                if not camera_images_path.is_dir():
                    continue
                    
                camera_images = camera_images_path.name
                camera_images_path_name.append(camera_images)
                img_list, video_list, label_video_path_list = self.get_img_video_path(input_dir, camera_images, camera_images_path)

                if 'depth' in camera_images:
                    # 处理深度图像
                    if img_list:
                        for img_path, video_path in zip(img_list, video_list):
                            logging.info(f"处理深度图像: {img_path} -> {video_path}")
                            if not self.encode_depth_video_frames(img_path, video_path, fps):
                                logging.warning(f"深度视频编码失败: {video_path}")
                else:
                    # 处理普通图像
                    if img_list:
                        for img_path, video_path in zip(img_list, video_list):
                            logging.info(f"处理普通图像(avi): {img_path} -> {video_path}")
                            if not self.encode_video_frames(img_path, video_path, fps):
                                logging.warning(f"普通视频编码失败: {video_path}")
                        for img_path, label_video_path in zip(img_list, label_video_path_list):
                            logging.info(f"处理标签图像(mp4): {img_path} -> {label_video_path}")
                            if not self.encode_label_video_frames(img_path, label_video_path, fps):
                                logging.warning(f"标签视频编码失败: {label_video_path}")

            # 更新元数据
            with open(meta_dir / "info.json", "r", encoding="utf-8") as f:
                metadata = json.load(f)
                
            if "features" not in metadata:
                raise ValueError("无效元数据: 缺少'features'字段")
                
            for field_name, field_info in metadata["features"].items():
                if field_name in camera_images_path_name:
                    field_info["dtype"] = "video"
                    
            metadata.update({
                "total_videos": 1,
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.avi",
                "image_path": None
            })
            
            with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
            # 删除原始images目录
            shutil.rmtree(images_dir)
            logging.info(f"已完成图像编译: {input_dir}")

    def _get_current_code_version(self) -> str:
        """获取当前代码版本"""
        try:
            return "1.0"
        except:
            return "unknown"

    def _check_data_version(self, data_path: Path) -> bool:
        """
        检查数据版本是否与代码兼容

        Args:
            data_path: 数据目录路径

        Returns:
            bool: 是否兼容
        """
        version_file = data_path / "meta" / "info.json"
        if not version_file.exists():
            return False
            
        with open(version_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            data_version = metadata.get("dorobot_dataset_version", "unknown")
            
        return data_version in self.supported_data_versions

    def _update_output_file_name(self, data_path: Path):
        """
        从common_record.json获取任务名称和ID，用于构建输出目录

        Args:
            data_path: 数据目录路径
        """
        task_file = data_path / "meta" / "common_record.json"
        if not task_file.exists():
            return False
            
        with open(task_file, "r", encoding="utf-8") as f:
            taskdata = json.load(f)
            task_id = taskdata.get("task_id", "unknown")
            task_name = taskdata.get("task_name", "unknown")
            output_name = f"{task_name}_{task_id}"
            self.output_dir = self.output_dir / output_name
            meta_dir = self.output_dir / "meta"
            if meta_dir.exists():
                raise FileExistsError(f"Error: 'meta' directory already exists in {self.output_dir / output_name}.")
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _update_code_from_repo(self):
        """从仓库自动更新代码"""
        if not self.repo_url:
            raise ValueError("Repository URL not provided for code update")
            
        try:
            logging.info(f"Updating code from repository: {self.repo_url}")
            subprocess.run(["git", "pull", self.repo_url], check=True)
            self.current_code_version = self._get_current_code_version()
            logging.info(f"Code updated to version: {self.current_code_version}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to update code from repository: {e}")

    def _validate_data_structure(self, data_path: Path) -> bool:
        """
        验证数据目录结构是否正确：
        1. 必须包含 `meta` 目录
        2. 必须包含 `images` 或 `videos` 目录（但不能同时存在）
        3. 所有必需目录必须非空

        Args:
            data_path: 要验证的数据目录路径

        Returns:
            bool: 结构是否有效
        """
        required_dirs = {"meta", "data"}
        optional_dirs = {"images", "videos"}
        existing_dirs = {d.name for d in data_path.iterdir() if d.is_dir()}
        self.existing_dirs_list.append(existing_dirs)
 
        # 检查必需目录是否存在
        if not required_dirs.issubset(existing_dirs):
            missing = required_dirs - existing_dirs
            logging.error(f"缺少必需目录 {missing} in {data_path}")
            return False
        
        # 检查 `images` 和 `videos` 是否互斥
        media_dirs = {d for d in existing_dirs if d in optional_dirs}
        if len(media_dirs) != 1:
            logging.error(f"必须且只能存在一个媒体目录（images/videos），当前找到: {media_dirs}")
            return False
 
        # 检查所有必需目录是否非空
        for dir_name in required_dirs | media_dirs:
            dir_path = data_path / dir_name
            if not any(dir_path.iterdir()):
                logging.error(f"目录 {dir_path} 为空")
                return False
 
        return True

    def _check_consistent_structure(self) -> bool:
        """
        检查所有输入目录的结构是否一致

        Returns:
            bool: 是否一致
        """
        if not self.existing_dirs_list:
            return False
        
        reference_structure = self.existing_dirs_list[0]
        reference_optional = reference_structure & {"images", "videos"}
        
        if len(reference_optional) != 1:
            logging.error(f"参考结构包含无效的可选目录: {reference_optional}")
            return False
        
        for structure in self.existing_dirs_list[1:]:
            current_optional = structure & {"images", "videos"}
            if current_optional != reference_optional:
                logging.error(f"目录不一致。期望: {reference_optional}, 找到: {current_optional}")
                return False
            
            required_dirs = {"meta", "data"}
            if not required_dirs.issubset(structure):
                missing = required_dirs - structure
                logging.error(f"某些文件夹缺少必需目录: {missing}")
                return False
        
        return True

    def _detect_file_extension(self, input_dir: Path) -> Optional[str]:
        """
        检测目录中episode文件的扩展名

        Args:
            input_dir: 要搜索的输入目录

        Returns:
            检测到的文件扩展名（如 ".mp4"），未找到返回None
        """
        for file in input_dir.rglob("episode_[0-9]*.*"):
            if file.is_file():
                return file.suffix.lower() or None
        return None

    def _get_middle_file_path(self, file_path: Path, sub: str) -> Path:
        """
        获取文件路径中指定子目录之后的路径部分

        Args:
            file_path: 完整文件路径
            sub: 要查找的子目录名

        Returns:
            Path对象，表示sub之后的路径（不包括文件名）
        """
        parts = file_path.parts
        try:
            sub_index = len(parts) - 1 - parts[::-1].index(sub)
        except ValueError:
            return Path()
        
        if sub_index < len(parts) - 1:
            middle_parts = parts[sub_index + 1:-1]
            return Path(*middle_parts) if middle_parts else Path()
        return Path()

    def _reindex_episode_files(self, input_dirs: List[Path], output_dir: Path, sub: str, file_extension: str):
        """
        重新编号episode文件并复制到输出目录

        Args:
            input_dirs: 输入目录列表
            output_dir: 输出根目录
            sub: 子目录名
            file_extension: 文件扩展名（如.mp4）
        """
        output_dir = output_dir / sub
        relative_path_list = []
        logging.info(f"检测到文件扩展名: {file_extension}")
        
        episode_counter = 0
        
        for input_dir in input_dirs:
            for root, _, files in os.walk(input_dir):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.suffix.lower() != file_extension:
                        continue
                    
                    # 跳过已正确命名的文件
                    if file.startswith("episode_") and file[8:].isdigit():
                        continue
                        
                    relative_path = self._get_middle_file_path(file_path, sub)
                    episode_counter = relative_path_list.count(relative_path)
                    relative_path_list.append(relative_path)
                    
                    new_filename = f"episode_{episode_counter:06d}{file_extension}"
                    output_path = output_dir / relative_path / new_filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(file_path, output_path)
        
        logging.info(f"成功重新编号 {episode_counter} 个文件到 {output_dir}")

    def _merge_jsonl_files(self, input_dirs: List[Path], output_dir: Path, key_field: Optional[str], sub: str):
        """
        合并JSONL文件

        Args:
            input_dirs: 输入目录列表
            output_dir: 输出目录
            key_field: 用于重新索引的字段（可选）
            sub: 子目录名
        """
        output_dir = output_dir / sub
        output_dir.mkdir(parents=True, exist_ok=True)
        
        jsonl_files = defaultdict(list)
        json_files = defaultdict(list)
        
        for input_dir in input_dirs:
            if not input_dir.exists():
                continue
        
            for file in input_dir.iterdir():
                if file.name == "info.json":
                    continue
        
                if file.suffix.lower() == ".jsonl":
                    jsonl_files[file.name].append(file)
                elif file.suffix.lower() == ".json":
                    json_files[file.name].append(file)
        
        # 合并JSONL文件
        for filename, files in jsonl_files.items():
            relative_path = self._get_middle_file_path(files[0], sub)
            output_path = output_dir / relative_path / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            records = []
            if key_field:
                temp_records = []
                for file in files:
                    with open(file, 'r', encoding='utf-8') as f:
                        for line in f:
                            record = json.loads(line.strip())
                            if key_field in record:
                                temp_records.append(record)
                
                if temp_records:
                    temp_records.sort(key=lambda x: x[key_field])
                    for i, record in enumerate(temp_records):
                        new_record = record.copy()
                        new_record[key_field] = i
                        records.append(new_record)
                    logging.info(f"  合并完成: {filename} (共 {len(records)} 条记录)")
                else:
                    for file in files:
                        with open(file, 'r', encoding='utf-8') as f:
                            for line in f:
                                records.append(json.loads(line.strip()))
                                logging.info(f"  仅保留首条记录:{filename}")
                                break
                        break
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # 合并JSON文件
        for filename, files in json_files.items():
            relative_path = self._get_middle_file_path(files[0], sub)
            output_path = output_dir / relative_path / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            merged_data = None
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data = data
                    # if isinstance(data, list):
                    #     merged_data.extend(data)
                    # else:
                    #     merged_data.append(data)
                    logging.info(f"  仅保留首条记录:{filename}")
                    break
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)

    def _merge_meta_info_files(self, input_dirs: List[Path], output_dir: Path, sub: str):
        """
        合并元数据信息文件

        Args:
            input_dirs: 输入目录列表
            output_dir: 输出目录
            sub: 子目录名
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = "info.json"
        episode_name = "episodes.jsonl"
        data_count = len(input_dirs)
        lengths = []
        
        for input_dir in input_dirs:
            info_path = input_dir / file_name
            if not info_path.exists():
                continue
                
            output_path = output_dir / "meta" / file_name
            episode_path = output_dir / "meta" / episode_name
            
            # 读取episode长度
            if episode_path.exists():
                with open(episode_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        data = json.loads(line.strip())
                        lengths.append(data["length"])
            
            # 计算统计信息
            total = sum(lengths) if lengths else 0
            shutil.copy2(info_path, output_path)
            
            with open(output_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                features = metadata["features"]
                count = sum(1 for key in features.keys() if key.startswith("observation.images"))
                
                metadata.update({
                    "total_videos": int(data_count * count),
                    "total_episodes": int(data_count),
                    "total_frames": int(total),
                    "splits": {"train": f"0:{str(data_count)}"}
                })
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            break

    @staticmethod
    def encode_video_frames(imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
        """
        编码普通视频帧（自动检测图片后缀）

        Args:
            imgs_dir: 包含图像序列的目录
            video_path: 输出视频路径
            fps: 帧率

        Returns:
            bool: 是否成功
        """
        try:
            imgs_dir = Path(imgs_dir)
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            
            supported_extensions = ['.jpg', '.jpeg', '.png']
            detected_ext = None
            for ext in supported_extensions:
                if list(imgs_dir.glob(f"*{ext}")):
                    detected_ext = ext
                    break
            
            if not detected_ext:
                raise ValueError(f"未找到支持的图片文件（支持: {', '.join(supported_extensions)}）")
            
            ffmpeg_args = OrderedDict([
                ("-f", "image2"),
                ("-r", str(fps)),
                ("-i", str(imgs_dir / f"frame_%06d{detected_ext}")),
                ("-vcodec", "libx264"),
                ("-pix_fmt", "yuv420p"),
                ("-g", "5"),
                ("-crf", "18"),
                ("-loglevel", "error"),
            ])
            
            ffmpeg_cmd = ["ffmpeg"] + [item for pair in ffmpeg_args.items() for item in pair] + [str(video_path)]
            subprocess.run(ffmpeg_cmd, check=True)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            logging.info(f"普通视频编码成功: {video_path}")
            return True
        except Exception as e:
            logging.error(f"普通视频编码失败: {str(e)}")            
            return False

    @staticmethod
    def encode_label_video_frames(imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
        """
        编码标签视频帧

        Args:
            imgs_dir: 包含图像序列的目录
            video_path: 输出视频路径
            fps: 帧率

        Returns:
            bool: 是否成功
        """
        try:
            imgs_dir = Path(imgs_dir)
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            
            supported_extensions = ['.jpg', '.jpeg', '.png']
            detected_ext = None
            for ext in supported_extensions:
                if list(imgs_dir.glob(f"*{ext}")):
                    detected_ext = ext
                    break
            
            if not detected_ext:
                raise ValueError(f"未找到支持的图片文件（支持: {', '.join(supported_extensions)}）")
            
            ffmpeg_args = OrderedDict([
                ("-f", "image2"),
                ("-r", str(fps)),
                ("-i", str(imgs_dir / f"frame_%06d{detected_ext}")),
                ("-vcodec", "libx264"),
                ("-pix_fmt", "yuv420p"),
                ("-g", "20"),
                ("-crf", "23"),
                ("-loglevel", "error"),
            ])
            
            ffmpeg_cmd = ["ffmpeg"] + [item for pair in ffmpeg_args.items() for item in pair] + [str(video_path)]
            subprocess.run(ffmpeg_cmd, check=True)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            logging.info(f"标签视频编码成功: {video_path}")
            return True
        except Exception as e:
            logging.error(f"标签视频编码失败: {str(e)}")
            return False

    @staticmethod
    def encode_depth_video_frames(imgs_dir: Union[Path, str], video_path: Union[Path, str], fps: int) -> bool:
        """
        编码深度视频帧

        Args:
            imgs_dir: 包含深度图像序列的目录
            video_path: 输出视频路径
            fps: 帧率

        Returns:
            bool: 是否成功
        """
        try:
            imgs_dir = Path(imgs_dir)
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)

            ffmpeg_args = [
                "ffmpeg",
                "-f", "image2",
                "-r", str(fps),
                "-i", str(imgs_dir / "frame_%06d.png"),
                "-vcodec", "ffv1",
                "-loglevel", "error",
                "-pix_fmt", "gray16le",
                "-y",
                str(video_path)
            ]
            
            subprocess.run(ffmpeg_args, check=True)
            
            if not video_path.exists():
                raise OSError(f"视频文件未生成: {video_path}")
                
            logging.info(f"深度视频编码成功: {video_path}")
            return True
        except Exception as e:
            logging.error(f"深度视频编码失败: {str(e)}")
            return False

    def prepare_merge(self, data_paths: List[Path]) -> bool:
        """
        合并前的准备工作

        Args:
            data_paths: 要合并的数据目录路径列表

        Returns:
            bool: 是否准备好进行合并

        Raises:
            ValueError: 数据版本不匹配或结构无效时抛出
            RuntimeError: 准备失败时抛出
        """
        logging.info(f"当前代码版本: {self.current_code_version}")
        # 1. 检查数据版本
        for path in data_paths:
            if not self._check_data_version(path):
                logging.error(f"数据版本检查失败: {path}")
                if self.repo_url:
                    logging.info("尝试从仓库更新代码...")
                    self._update_code_from_repo()
                    if not self._check_data_version(path):
                        raise ValueError(f"数据版本不匹配且更新失败: {path}")
                else:
                    raise ValueError(f"数据版本不匹配且未提供仓库URL: {path}")
        # 2. 检查数据结构
        for path in data_paths:
            if not self._validate_data_structure(path):
                raise ValueError(f"无效的数据结构: {path}")
        # 3. 检查所有目录结构是否一致      
        if not self._check_consistent_structure():
            raise ValueError("输入文件夹的目录结构不一致")
        # 4.更新输出目录名称   
        for path in data_paths:            
            self._update_output_file_name(path)
            break
        
        return True

    def merge(self):
        """执行主合并流程"""
        try:
            # 分类数据目录
            data_compile_list = [d for d in self.data_dirs if (d / "images").exists()]

            # 1. 编译图像数据为视频格式
            if data_compile_list:
                logging.info(f"正在编译 {len(data_compile_list)} 个图像数据集为视频格式...")
                self._compile_images_to_videos(data_compile_list)
                logging.info("编译完成")

            if not self.prepare_merge(self.data_dirs):
                raise RuntimeError("合并准备阶段失败")
            
            # 2. 合并所有数据
            for sub in self.existing_dirs_list[0]:
                has_episode_file = any(self._detect_file_extension(d / sub) for d in self.data_dirs)
                logging.info("")
                if has_episode_file:
                    logging.info(f"合并 {sub} 中的视频文件...")
                    new_data_dirs = [d / sub for d in self.data_dirs]
                    suffix = self._detect_file_extension(new_data_dirs[0])
                    self._reindex_episode_files(new_data_dirs, self.output_dir, sub, suffix)
                else:
                    logging.info(f"合并 {sub} 中的元数据文件...")
                    new_data_dirs = [d / sub for d in self.data_dirs]
                    self._merge_jsonl_files(new_data_dirs, self.output_dir, "episode_index", sub)
                    self._merge_meta_info_files(new_data_dirs, self.output_dir, sub)
            
            logging.info(f"合并完成！结果保存至: {self.output_dir}")
        except Exception as e:
            logging.error(f"合并过程中发生错误: {str(e)}")
            raise


# 使用示例
if __name__ == "__main__":
    data_dirs = "/home/liuyou/Documents/wanx-studio-dataset-example/20250905/dev/加热三明治_liuyou_test_220/"
    output_dir = "/home/liuyou/dorobot_merged_output/"
    merger = DoRobotDataMerger(
        repo_url="git@github.com:liuyou1103/any_to_lerobot.git",
        data_dirs=data_dirs, # 需合并数据目录
        output_dir=output_dir, # 输出目录
        log_to_file=True  # 启用日志文件
        
    )
    
    try:
        merger.merge()
    except Exception as e:
        logging.error(f"合并失败: {e}")

import itertools
import json
import os
import numpy as np
import shutil
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import datetime
import zipfile
try:
    # v2.1
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.1'
except:
    # v2.0
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    _LEROBOT_VERSION = '2.0'

from utils.data_tools import eef_states_to_pos_rot_grip, pos_rot_grip_to_eef_states
from utils.image_tools import load_image
from utils.transforms import compute_relative_pose, robust_compute_relative_pose
from nas_sdk import NASAuthenticator
from pika_config import DataConfig

# 获取 LeRobot 数据集的默认根目录
def get_lerobot_default_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')

def load_sync(file_path):
    """加载同步文件"""
    with open(file_path, 'r') as f:
        filenames = f.readlines()
    return [filename.strip() for filename in filenames]

class DataStatusRecorder:
    def __init__(self, status_file_path):
        self.status_file = status_file_path
        self.status_data = {
            'processed_items': {},  # 格式: {json_path: {data_path: "status", ...}}
            'task_summary': {},     # 格式: {task_name: {"completed": bool, "timestamp": str}}
            'last_run': None
        }
        self._load_status()
    
    def _load_status(self):
        """加载状态文件"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:  # 添加编码参数
                    self.status_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"状态文件加载失败，将重新初始化: {str(e)}")
                self.status_data = {
                    'processed_items': {},
                    'task_summary': {},
                    'last_run': None
                }
    
    def save_status(self):
        """保存状态到文件"""
        try:
            self.status_data['last_run'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.status_file, 'w', encoding='utf-8') as f:  # 添加编码参数
                json.dump(self.status_data, f, indent=2, ensure_ascii=False)  # 关键修改
        except IOError as e:
            print(f"保存状态文件失败: {str(e)}")
    
    def is_data_processed(self, json_path, data_path):
        """检查特定数据是否已处理"""
        return json_path in self.status_data['processed_items'] and \
               data_path in self.status_data['processed_items'][json_path] and \
               self.status_data['processed_items'][json_path][data_path] == 'completed'
    
    def mark_data_completed(self, json_path, data_path, task_name):
        """标记数据为已处理"""
        if json_path not in self.status_data['processed_items']:
            self.status_data['processed_items'][json_path] = {}
        
        self.status_data['processed_items'][json_path][data_path] = 'completed'
        
        # 更新任务摘要
        self.status_data['task_summary'][task_name] = {
            'completed': True,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_count': len(self.status_data['processed_items'][json_path])
        }
        self.save_status()
    
    def get_task_status(self, task_name):
        """获取任务状态"""
        return self.status_data['task_summary'].get(task_name, {'completed': False})
    
    def mark_data_failed(self, json_path, data_path, task_name, error_msg):
        """标记数据为处理失败"""
        if json_path not in self.status_data['processed_items']:
            self.status_data['processed_items'][json_path] = {}
        
        # 记录失败状态和错误信息
        self.status_data['processed_items'][json_path][data_path] = {
            'status': 'failed',
            'error': error_msg,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_status()
    
class PikaDataProcessor(object):
    def __init__(self, config):
        self.config = config
        print(f"Configuration: {self.config}")
        self.current_json_files = []  # 新增：存储当前处理的JSON文件路径
        self.path_to_data_info = {}
        self.skipped_data_paths = []  # 新增：存储跳过的data_path
        self.skip_log_path = '/home/ctos/pika/skip'  
        self.path_to_json_item = {}  # {path: (order, json_item)}
        self.path_info = {}          # {path: order}（
        
        status_dir = '/home/ctos/pika/status'
        os.makedirs(status_dir, exist_ok=True)
        self.status_recorder = DataStatusRecorder(
            os.path.join(status_dir, 'data_processing_status.json')
        )
        
        
        if self.config.overwrite:
            if self.config.data_root is not None:
                data_root = self.config.data_root
                if os.path.exists(data_root):
                    print(f'Overwriting data root: {data_root}? (y/n)', end=' ')
                    if input().strip().lower() != 'y':
                        print('Exiting without overwriting.')
                        return
                    shutil.rmtree(data_root, ignore_errors=True)
            else:
                data_root = get_lerobot_default_root()
                data_root = os.path.join(data_root, self.config.repo_id)
                if os.path.exists(data_root):
                    print(f'Overwriting data root: {data_root}? (y/n)', end=' ')
                    if input().strip().lower() != 'y':
                        print('Exiting without overwriting.')
                        return
                    shutil.rmtree(data_root, ignore_errors=True)
        
       
        os.makedirs(self.skip_log_path, exist_ok=True)
        
        self.create_dataset()
        self.nas_auth = NASAuthenticator()
        if not self.nas_auth.get_auth_sid():
            raise Exception("NAS 认证失败")

    def get_today_time(self):
        # 获取当前日期和时间
        today = datetime.datetime.now()
        # 格式化日期为字符串，格式为 "YYYYMMDDHHMMSS"
        date_string = today.strftime("%Y%m%d%H%M%S")
        print(f"Today's time: {date_string}")
        return date_string

    # 新增：保存跳过的data_path到CSV文件
    def save_skipped_paths(self):
        if not self.skipped_data_paths:
            print("没有需要跳过的文件记录")
            return
            
        # 使用当前日期作为文件名
        today = datetime.datetime.now().strftime("%Y%m%d")
        skip_file = os.path.join(self.skip_log_path, f"skipped_{today}.csv")
        
        # 创建DataFrame并保存
        df = pd.DataFrame({"data_path": self.skipped_data_paths})
        # 如果文件已存在，则追加；否则创建新文件
        if os.path.exists(skip_file):
            df.to_csv(skip_file, mode='a', header=False, index=False)
        else:
            df.to_csv(skip_file, index=False)
        
        print(f"已将 {len(self.skipped_data_paths)} 条跳过的记录保存到 {skip_file}")

    def create_dataset(self):
        """创建数据集结构"""
        if getattr(self.config, 'check_only', False):
            print('Check only mode, skipping dataset creation.')
            return
        
        # 定义 RGB 图像的配置信息
        rgb_config = {
            'dtype': 'video',
            'shape': (self.config.image_height, self.config.image_width, 3),
            'names': ['height', 'width', 'channels'],
        }
        # 根据配置中的 RGB 名称列表，为每个 RGB 名称创建相同的配置
        features = {rgb_name: rgb_config for rgb_name in self.config.rgb_names}
        print(f"RGB features: {features}")
        
        # 计算动作的总长度
        action_len = sum(len(keys) for keys in self.config.action_keys_list_2)
        # 将动作键列表展平为一维列表
        action_keys_flatten = list(itertools.chain.from_iterable(self.config.action_keys_list_2))
        # 定义动作的配置信息
        features[self.config.action_name] = {
            'dtype': 'float64',
            'shape': (action_len,),
            'names': action_keys_flatten,
        }
        print(f"Action feature: {features[self.config.action_name]}")

        # 如果配置中使用状态信息
        if getattr(self.config, 'use_state', False):
            # 定义状态的配置信息，与动作配置相同
            features[self.config.state_name] = {
                'dtype': 'float64',
                'shape': (action_len,),
                'names': action_keys_flatten,
            }
            print(f"State feature: {features[self.config.state_name]}")
        
        # 如果配置中使用深度信息
        if getattr(self.config, 'use_depth', False):
            # 定义深度图像的配置信息
            depth_config = {
                'dtype': 'video',
                'shape': (self.config.image_height, self.config.image_width),
                'name': ['height', 'width'],
            }
            # 根据配置中的深度名称列表，为每个深度名称创建相同的配置
            for depth_name in self.config.depth_names:
                features[depth_name] = depth_config
            print(f"Depth features: {features}")
        
        
        lerobot_base = '/home/ctos/pika/convert/'
        # 直接使用基础目录作为数据根目录，不再添加时间戳
        self.config.data_root = lerobot_base
        print(f"Data root: {self.config.data_root}")
        
        self.dataset = None

    def _extract_task_name(self, path):
        """从路径中提取task_name（不区分pika的大小写）"""
        parts = path.split('/')
        # 将路径部分统一转换为小写进行比较
        lower_parts = [part.lower() for part in parts]
        try:
            # 查找不区分大小写的"pika"
            pika_index = lower_parts.index('pika')
            if pika_index + 1 < len(parts):
                # 返回原始大小写的任务名称
                return parts[pika_index + 1]
        except ValueError:
            pass
        return "unknown_task"
    
    def _init_label_folder(self, task_data_root):
        """初始化标签目录和标注文件"""
        # 构建标签文件夹的完整路径，在任务数据根目录下创建名为'label'的子文件夹
        label_dir = os.path.join(task_data_root, 'label')
        # 创建标签文件夹，如果文件夹已存在则不做任何操作（exist_ok=True确保不会抛出异常）
        os.makedirs(label_dir, exist_ok=True)
        # 构建标注文件的完整路径，标注文件名为'data_annotation.json'，位于label文件夹下
        annotation_file = os.path.join(label_dir, 'data_annotation.json')
        
        if not os.path.exists(annotation_file):
            with open(annotation_file, 'w') as f:
                json.dump([], f)

        abnormal_file = os.path.join(task_data_root, 'label', 'Abnormal.json')
        os.makedirs(os.path.dirname(abnormal_file), exist_ok=True)

        # 初始化异常记录数据结构
        abnormal_data = {}
        if os.path.exists(abnormal_file):
            with open(abnormal_file, 'r') as f:
                try:
                    abnormal_data = json.load(f)
                except json.JSONDecodeError:
                    abnormal_data = {}

                # 返回标注文件的完整路径，供调用者使用
        return annotation_file,abnormal_file, abnormal_data

    def _backup_json_file(self, json_file, task_name):
        """将原始JSON文件备份到label目录"""
        task_data_root = os.path.join(self.config.data_root, task_name)
        label_dir = os.path.join(task_data_root, 'label')
        os.makedirs(label_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"raw_{timestamp}_{os.path.basename(json_file)}"
        shutil.copy2(json_file, os.path.join(label_dir, new_name))
   

    def process_data(self):
        json_dir = '/home/ctos/lerobot/src/pika/json'
        csv_dir = '/home/ctos/lerobot/src/pika/csv'
        local_base_dir = '/home/ctos/pika/nas'

        self.current_json_files = []
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    self.current_json_files.append(json_path)
                    print(f"Found JSON file: {json_path}")
        # 检查JSON目录是否存在，不存在则报错并返回
        print(f"Checking JSON directory: {json_dir}")
        if not os.path.exists(json_dir):
            print(f"Error: JSON directory {json_dir} does not exist!")
            return

        # 列出所有找到的 JSON 文件
        json_files = []
        # 递归遍历JSON目录及其子目录，筛选出所有.json文件
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        print(f"Found {len(json_files)} JSON files in {json_dir}")
        # 打印所有找到的JSON文件路径
        for file_path in json_files:
            print(f"  - {file_path}")
        
        # 提取所有 JSON 文件中的 path 并记录顺序
        path_info = {}  # {path: order}
        for file_path in json_files:
            print(f"Processing JSON file: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"  JSON文件包含数组，有{len(data)}个元素")
                        for idx, item in enumerate(data, start=1):
                            if isinstance(item, dict) and 'path' in item:
                                path = item['path']
                                path_info[path] = idx
                                # 新增：存储完整 JSON 数据项
                                self.path_to_json_item[path] = (idx, item)
                                print(f"  找到路径: {path} (顺序: {idx})")
                    elif isinstance(data, dict):
                        print(f"  JSON结构键: {list(data.keys())}")
                        if 'path' in data:
                            path = data['path']
                            path_info[path] = 1
                            # 新增：存储完整 JSON 数据项
                            self.path_to_json_item[path] = (1, data)
                            print(f"  找到路径: {path} (顺序: 1)")
                    else:
                        print("  不支持的JSON格式")
            except json.JSONDecodeError as e:
                print(f"解析JSON文件{file_path}出错: {e}")

        # 将 path_info 保存到成员变量
        self.path_info = path_info
        
        
        # 读取所有 csv 文件，找到对应的 data_path 并提取 task_name
        path_to_data_info = {}  # {path: (data_path, task_name)}
        tasks_names = set()  # 用于存储所有唯一的 tasks_name
        # 遍历CSV目录及其子目录，处理所有.csv文件
        for root, dirs, files in os.walk(csv_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"Processing CSV file: {file_path}")
                    try:
                        df = pd.read_csv(file_path, sep=',')
                        print(f"  CSV columns: {list(df.columns)}")
                        print(f"  CSV rows: {len(df)}")
                        
                        if 'json_path' not in df.columns or 'data_path' not in df.columns:
                            print(f"  Warning: 'json_path' or 'data_path' column not found in {file_path}")
                            continue
                        
                        for path in path_info.keys():
                            matching_rows = df[df['json_path'] == path]
                            if not matching_rows.empty:
                                data_path = matching_rows['data_path'].values[0]
                                task_name = self._extract_task_name(data_path)
                                
                                # 检查是否已处理过
                                if self.status_recorder.is_data_processed(path, data_path):
                                    print(f"数据已处理，跳过 - JSON路径: {path}, NAS路径: {data_path}")
                                    continue
                                    
                                path_to_data_info[path] = (data_path, task_name)
                                tasks_names.add(task_name)
                                print(f"[CSV映射] json_path={path} -> data_path={data_path} (task: {task_name})")

                                # 备份JSON文件（原有逻辑不变）
                                for json_file in self.current_json_files:
                                    if path in json_file:
                                        self._backup_json_file(json_file, task_name)
                                        break



                                for json_file in self.current_json_files:
                                    if path in json_file:
                                        self._backup_json_file(json_file, task_name)                               
                        
                    except Exception as e:
                        print(f"Error processing CSV file {file_path}: {e}")

        print(f"Found {len(path_to_data_info)} data paths from CSV files.")
        print(f"Found tasks_names: {tasks_names}")
        self.path_to_data_info = path_to_data_info

        
        
        if not path_to_data_info:
            print("No data paths found. Exiting without further processing.")
            # 保存跳过记录（如果有的话）
            self.save_skipped_paths()
            return

        # 为每个 tasks_name 创建数据集
        for task_name in tasks_names:
            # 创建任务特定的数据根目录
            task_data_root = os.path.join(self.config.data_root, task_name)
            
            # 创建任务特定的数据集
            self._create_task_dataset(task_data_root)
            
            # 处理属于当前 task_name 的数据路径
            task_paths = [path for path, (_, tn) in path_to_data_info.items() if tn == task_name]
            print(f"\n[任务 {task_name}] 共找到 {len(task_paths)} 条匹配路径")
            # 新增：打印当前任务的有效order列表
            valid_orders = [self.path_to_json_item[path][0] for path in task_paths if path in self.path_to_json_item]
            print(f"[任务 {task_name}] 有效order列表: {sorted(valid_orders)}")
            
            # 下载并处理每个数据文件
            for path in task_paths:
                data_path, _ = path_to_data_info.get(path)
                if data_path is None:
                    print(f"Could not find data_path for {path}")
                    continue
                
                order = path_info[path]
                
                # 创建任务特定的本地目录
                task_local_dir = os.path.join(local_base_dir, task_name)
                os.makedirs(task_local_dir, exist_ok=True)
                
                # 构建本地路径
                local_path = os.path.join(task_local_dir, f"episode{order}.zip")
                print(f"Data path: {data_path} -> Local path: {local_path} (顺序: {order})")

                try:  
                    # 下载文件
                    print(f"Downloading {data_path} to {local_path}")
                    if not self.nas_auth.download_file(data_path, local_path):
                        print(f"  Download failed for {data_path}")
                        # 下载失败，添加到跳过列表
                        self.skipped_data_paths.append(data_path)
                        continue

                    print(f"  Download successful")
                    
                    # 创建解压目录
                    extract_dir = os.path.join(task_local_dir, str(order))
                    os.makedirs(extract_dir, exist_ok=True)

                    try:
                        # 尝试以 ZIP 文件打开
                        with zipfile.ZipFile(local_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                            print(f"  Extracted {local_path} to {extract_dir}")
                            
                            # 重命名 episode 文件夹为顺序编号
                            for root_dir, dirs, files in os.walk(extract_dir):
                                for dir_name in dirs:
                                    if dir_name.startswith('episode'):
                                        old_path = os.path.join(root_dir, dir_name)
                                        new_path = os.path.join(root_dir, f"episode{order}")
                                        os.rename(old_path, new_path)
                                        print(f"  Renamed {old_path} to {new_path}")
                                        break
                            
                            # 删除压缩包
                            os.remove(local_path)
                            print(f"  Deleted {local_path}")
                    except zipfile.BadZipFile:
                        print(f"  File {local_path} is not a valid zip file")
                        # 解压失败，添加到跳过列表
                        self.skipped_data_paths.append(data_path)
                        continue

                    # 处理下载的数据
                    episode_dir = os.path.join(extract_dir, f"episode{order}")
                    if not os.path.exists(episode_dir):
                        print(f"Episode directory {episode_dir} not found after extraction")
                        # 目录不存在，添加到跳过列表
                        self.skipped_data_paths.append(data_path)
                        continue

                    print(f'Processing source 1/1, episode {order}: {episode_dir}')
                    # 处理episode，捕获可能的异常
                    try:
                        self._add_episode(episode_dir, task_name)
                        
                        # 处理成功，更新状态
                        self.status_recorder.mark_data_completed(path, data_path, task_name)
                        print(f"数据处理完成并记录状态 - JSON路径: {path}, NAS路径: {data_path}")
                        
                    except Exception as e:
                        print(f"  Error processing episode {episode_dir}: {str(e)}")
                        self.skipped_data_paths.append(data_path)
                        self.status_recorder.mark_data_failed(path, data_path, task_name, str(e))
                        continue

                except Exception as e:
                    print(f"  Unexpected error processing {data_path}: {str(e)}")
                    # 发生意外错误，添加到跳过列表
                    self.skipped_data_paths.append(data_path)
                    continue

        # 处理完成后保存跳过的记录
        self.save_skipped_paths()

    def _create_task_dataset(self, task_data_root):
        """为特定任务创建数据集"""
        # 定义 RGB 图像的配置信息
        rgb_config = {
            'dtype': 'video',
            'shape': (self.config.image_height, self.config.image_width, 3),
            'names': ['height', 'width', 'channels'],
        }
        # 根据配置中的 RGB 名称列表，为每个 RGB 名称创建相同的配置
        features = {rgb_name: rgb_config for rgb_name in self.config.rgb_names}
        
        # 计算动作的总长度
        action_len = sum(len(keys) for keys in self.config.action_keys_list_2)
        # 将动作键列表展平为一维列表
        action_keys_flatten = list(itertools.chain.from_iterable(self.config.action_keys_list_2))
        # 定义动作的配置信息
        features[self.config.action_name] = {
            'dtype': 'float64',
            'shape': (action_len,),
            'names': action_keys_flatten,
        }

        # 如果配置中使用状态信息
        if getattr(self.config, 'use_state', False):
            # 定义状态的配置信息，与动作配置相同
            features[self.config.state_name] = {
                'dtype': 'float64',
                'shape': (action_len,),
                'names': action_keys_flatten,
            }
        
        # 如果配置中使用深度信息
        if getattr(self.config, 'use_depth', False):
            # 定义深度图像的配置信息
            depth_config = {
                'dtype': 'video',
                'shape': (self.config.image_height, self.config.image_width,1),
                'name': ['height', 'width','channel'],
                'is_depth':True
            }
            # 根据配置中的深度名称列表，为每个深度名称创建相同的配置
            for depth_name in self.config.depth_names:
                features[depth_name] = depth_config
        
        # 使用 LeRobotDataset 创建数据集
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            root=task_data_root,
            fps=self.config.fps,
            video_backend=getattr(self.config, 'video_backend', 'pyav'),
            features=features,
            image_writer_processes = 0,
            image_writer_threads = 4,
        )
        print(f"Dataset created successfully for task at {task_data_root}")

# 在 _add_episode 方法中修改参数和保存逻辑
    def _add_episode(self, episode_path, task_name):
        # 构建当前任务的数据根目录路径（数据根目录/任务名称）
        task_data_root = os.path.join(self.config.data_root, task_name)
        # 初始化标签文件夹并获取标注文件路径（确保标注文件存在）
        annotation_file, abnormal_file, abnormal_data = self._init_label_folder(task_data_root)

        # 创建device目录和相关文件
        device_dir = os.path.join(task_data_root, 'device')
        os.makedirs(device_dir, exist_ok=True)
        
        # 创建camera_intrinsic.json
        camera_intrinsic_file = os.path.join(device_dir, 'camera_intrinsic.json')
        if not os.path.exists(camera_intrinsic_file):
            with open(camera_intrinsic_file, 'w') as f:
                json.dump(self.config.camera_info, f, indent=2)
        
        # 创建device_info.json
        device_info_file = os.path.join(device_dir, 'device_info.json')
        if not os.path.exists(device_info_file):
            with open(device_info_file, 'w') as f:
                json.dump(self.config.device_info, f, indent=2)
        
        # 处理device_episode.json
        device_episode_file = os.path.join(device_dir, 'device_episode.json')
        try:
            episode_num = int(os.path.basename(episode_path).replace('episode', ''))
            episode_index = episode_num - 1
        except ValueError:
            episode_index = 0  # 默认值
        
        episode_device = {
            "episode_index": episode_index,
            "device_id": "9xK992pQ"
        }
        
        # 读取已有内容或初始化空列表
        existing_episodes = []
        if os.path.exists(device_episode_file):
            with open(device_episode_file, 'r') as f:
                try:
                    existing_episodes = json.load(f)
                except json.JSONDecodeError:
                    existing_episodes = []
        
        # 更新或添加episode记录
        existing_index = next((i for i, x in enumerate(existing_episodes) 
                            if x.get('episode_index') == episode_index), None)
        
        if existing_index is not None:
            existing_episodes[existing_index] = episode_device
        else:
            existing_episodes.append(episode_device)
        
        with open(device_episode_file, 'w') as f:
            json.dump(existing_episodes, f, indent=2)        
        
        # 加载已有标注
        existing_annotations = []
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                try:
                    existing_annotations = json.load(f)
                except json.JSONDecodeError:
                    existing_annotations = []
        
        raw_outputs = self._load_episode(episode_path)
        
        # 查找匹配的原始path和JSON数据
        print("\n===== DEBUG START =====")
        print(f"[当前处理的EPISODE路径] {episode_path}")
        print(f"[任务名称] {task_name}")

        try:
            order = int(os.path.basename(os.path.dirname(episode_path)))  # 如 ".../1/episode1" -> order=1
            print(f"[提取的顺序编号] order={order} (来自路径: {os.path.dirname(episode_path)})")
            episode_index = order - 1
        except (ValueError, IndexError):
            print(f"[错误] 无法从路径提取顺序编号: {episode_path}")
            episode_index = 0

        # 通过 order 查找匹配的 path 和 json_item
        matched_path = None
        matched_json_item = None
        nas_path = "unknown"
        # 先过滤出当前任务的所有路径
        task_candidate_paths = [
            path for path in self.path_to_json_item 
            if self.path_to_data_info.get(path, (None, None))[1] == task_name
        ]
        print(f"[匹配候选] 任务 {task_name} 下的候选路径数: {len(task_candidate_paths)}")

        for path in task_candidate_paths:
            item_order, json_item = self.path_to_json_item[path]
            if item_order == order:
                matched_path = path
                matched_json_item = json_item
                nas_path = self.path_to_data_info[path][0]  # 获取当前任务的NAS路径
                break

        # 调试输出候选路径的order分布
        candidate_orders = [self.path_to_json_item[path][0] for path in task_candidate_paths]
        print(f"[匹配候选] 任务 {task_name} 的order分布: {sorted(candidate_orders)}")
        if matched_json_item:
            print(f"✅ 匹配成功: order={order}")
            print(f"  - 原始路径: {matched_path}")
            print(f"  - NAS路径: {nas_path}")
            print(f"  - 标注内容: {matched_json_item.keys()}")
        else:
            print(f"❌ 匹配失败: 任务 {task_name} 中未找到order={order}对应的数据")
            print(f"  当前任务有效order: {sorted(candidate_orders)}")
            print(f"  期望匹配order: {order}")
            print(f"  请检查JSON文件中的order是否与CSV中的数据路径对应")


        # 更新标注文件
        if matched_json_item:
            matched_json_item['nas_path'] = nas_path
            new_annotation = {
                'episode_index': episode_index,
                # 'original_path': matched_path,
                'annotation': matched_json_item,
            }
            # 检查是否已存在该 episode_index 的记录
            existing_index = next((i for i, x in enumerate(existing_annotations) 
                                if x.get('episode_index') == episode_index), None)
            if existing_index is not None:
                existing_annotations[existing_index] = new_annotation
            else:
                existing_annotations.append(new_annotation)
            # 保存到文件
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(existing_annotations, f, indent=2, ensure_ascii=False)
            print(f"Updated annotations for episode {episode_index} in {annotation_file}")
        
        # 其余处理逻辑保持不变...
        if getattr(self.config, 'check_only', False):
            print(f'Check only mode, skipping adding episode {episode_path}')
            return

        raw_images = raw_outputs['raw_images']
        raw_actions = raw_outputs['raw_actions']
        instruction = raw_outputs['instruction']
        
        if getattr(self.config, 'use_depth', False):
            raw_depths = raw_outputs['raw_depths']

        indexs = list(range(len(raw_images[self.config.rgb_names[0]])))
        
        # 初始化当前episode的异常记录
        episode_key = f"episode_{episode_index}"
        current_episode_abnormal = {
            "total_frames": len(indexs[:-1]),
            "abnormal_frames_count": 0,
            "abnormal_percentage": "0.00%",
            "abnormal_frames": []  # 简化后的异常帧记录
        }

        for i in tqdm(indexs[:-1], desc=f'Adding episode {episode_path}'):
            states = np.concatenate([raw_actions[action_dir][i] for action_dir in self.config.action_dirs])
            actions = np.concatenate([raw_actions[action_dir][i + 1] for action_dir in self.config.action_dirs])
            
            # 检查是否是非空操作
            abnormal_frames = self._check_nonoop_actions(states, actions, i, episode_index)
            
            # 记录异常帧
            if abnormal_frames:
                current_episode_abnormal["abnormal_frames"].extend(abnormal_frames)
            
            # 构建帧数据（原有逻辑不变）
            frame = {}
            for rgb_name in self.config.rgb_names:
                frame[rgb_name] = load_image(raw_images[rgb_name][i])
            
            frame[self.config.action_name] = actions

            if getattr(self.config, 'use_state', False):
                frame[self.config.state_name] = states

            if getattr(self.config, 'use_depth', False):
                for depth_name in self.config.depth_names:
                    frame[depth_name] = img = load_image(raw_depths[depth_name][i])

            # 添加帧到数据集（原有逻辑不变）
            if _LEROBOT_VERSION == '2.0':
                self.dataset.add_frame(frame)
            elif _LEROBOT_VERSION == '2.1':
                self.dataset.add_frame(frame, task=task_name)  

        # 计算异常帧统计信息
        current_episode_abnormal["abnormal_frames_count"] = len(current_episode_abnormal["abnormal_frames"])
        current_episode_abnormal["abnormal_percentage"] = f"{current_episode_abnormal['abnormal_frames_count'] / current_episode_abnormal['total_frames'] * 100:.2f}%"
                # 更新异常数据
        abnormal_data[episode_key] = current_episode_abnormal

        # 保存异常数据到文件
        with open(abnormal_file, 'w') as f:
            json.dump(abnormal_data, f, indent=2)
        
        if _LEROBOT_VERSION == '2.0':
            self.dataset.save_episode(task=task_name)
        elif _LEROBOT_VERSION == '2.1':
            self.dataset.save_episode()
        else:
            raise ValueError(f'Unsupported LeRobot version: {_LEROBOT_VERSION}')
        print(f"Episode {episode_path} saved successfully with task name: {task_name}.")

    def _load_episode(self, episode_path):
        """加载单个episode数据"""
        raw_images = defaultdict(list)
        for rgb_dir, rgb_name in zip(self.config.rgb_dirs, self.config.rgb_names):
            rgb_dir = os.path.join(episode_path, rgb_dir)
            
            if os.path.exists(os.path.join(rgb_dir, 'sync.txt')):
                filenames = load_sync(os.path.join(rgb_dir, 'sync.txt'))
            else:
                filenames = os.listdir(rgb_dir)
                filenames = [f for f in filenames if f.endswith(('.jpg', '.png'))]
                filenames.sort(key=lambda x: float(x[:-4]))

            for filename in filenames:
                raw_images[rgb_name].append(os.path.join(rgb_dir, filename))
            print(f"Loaded {len(raw_images[rgb_name])} RGB images for {rgb_name} in {episode_path}")
        
        raw_actions = defaultdict(list)
        for action_dir, action_keys in zip(self.config.action_dirs, self.config.action_keys_list):
            action_dir_ = os.path.join(episode_path, action_dir)

            if os.path.exists(os.path.join(action_dir_, 'sync.txt')):
                filenames = load_sync(os.path.join(action_dir_, 'sync.txt'))
            else:
                filenames = os.listdir(action_dir_)
                filenames = [f for f in filenames if f.endswith('.json')]
                filenames.sort(key=lambda x: float(x[:-5]))

            for filename in filenames:
                action_path = os.path.join(action_dir_, filename)
                with open(action_path, 'r') as f:
                    action_data = json.load(f)
                action_data = np.array([action_data[key] for key in action_keys])
                raw_actions[action_dir].append(action_data)
            print(f"Loaded {len(raw_actions[action_dir])} actions for {action_dir} in {episode_path}")
        
        instruction_path = os.path.join(episode_path, self.config.instruction_path)
        with open(instruction_path, 'r') as f:
            instruction_data = json.load(f)
        
        instruction = instruction_data['instructions'][0]
        if instruction == 'null':
            instruction = self.config.default_instruction
        print(f"Instruction for {episode_path}: {instruction}")
        
        outputs = {
            'raw_images': raw_images,
            'raw_actions': raw_actions,
            'instruction': instruction
        }

        if getattr(self.config, 'use_depth', False):
            raw_depths = defaultdict(list)
            for depth_dir, depth_name in zip(self.config.depth_dirs, self.config.depth_names):
                depth_dir = os.path.join(episode_path, depth_dir)

                if os.path.exists(os.path.join(depth_dir, 'sync.txt')):
                    filenames = load_sync(os.path.join(depth_dir, 'sync.txt'))
                else:
                    filenames = os.listdir(depth_dir)
                    filenames = [f for f in filenames if f.endswith(('.png', '.jpg'))]
                    filenames.sort(key=lambda x: float(x[:-4]))
                
                for filename in filenames:
                    depth_path = os.path.join(depth_dir, filename)
                    raw_depths[depth_name].append(depth_path)
                print(f"Loaded {len(raw_depths[depth_name])} depth images for {depth_name} in {episode_path}")
            outputs['raw_depths'] = raw_depths
        
        # 验证所有数据长度一致
        lens = []
        for rgb_name, images_list in raw_images.items():
            lens.append(len(images_list))
        for action_dir, actions_list in raw_actions.items():
            lens.append(len(actions_list))
        if getattr(self.config, 'use_depth', False):
            for depth_name, depth_list in raw_depths.items():
                lens.append(len(depth_list))
        
        assert all(lens[0] == l for l in lens), "All lists must have the same length"
        
        return outputs

    def _check_nonoop_actions(self, states, actions, frame_index, episode_index):
        """检查非空操作并记录异常帧（所有状态变化都在阈值内则判定为异常）"""
        states = eef_states_to_pos_rot_grip(states)
        actions = eef_states_to_pos_rot_grip(actions)
        
        is_abnormal = True  # 初始假设是异常帧
        
        for state, action in zip(states, actions):
            position_diff = np.linalg.norm(np.array(state[0]) - np.array(action[0]))
            rotation_diff = np.linalg.norm(np.array(state[1]) - np.array(action[1]))
            
            # 如果任一变化超过阈值，则不是异常帧
            if (position_diff > self.config.position_nonoop_threshold or
                rotation_diff > self.config.rotation_nonoop_threshold):
                is_abnormal = False
                break
        
        # 如果是异常帧，返回帧号；否则返回空列表
        return [str(frame_index)] if is_abnormal else []

    def check_task_completion(self, task_name):
        """检查任务是否全部完成并打印状态"""
        task_status = self.status_recorder.get_task_status(task_name)
        if task_status.get('completed', False):
            print(f"任务 {task_name} 已全部完成 (最后处理时间: {task_status['timestamp']})")
            print(f"已处理数据项: {task_status.get('data_count', 0)}")
            return True
        else:
            print(f"任务 {task_name} 未全部完成或从未处理")
            return False   

    

 
def main():
    config = DataConfig()
    processor = PikaDataProcessor(config)
    processor.process_data()
    #processor._create_task_dataset('/home/ctos/pika/convert/task_fold_pants_1000_4.31')
    #processor._add_episode('/home/ctos/pika/nas/task_fold_pants_1000_4.31/1/episode1','task_fold_pants')

if __name__ == "__main__":
    main()
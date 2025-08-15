import json
import os
import numpy as np
import shutil
from collections import defaultdict
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import cv2

from aloha_config import DataConfig
import h5py
import datetime


class DataProcessor(object):
    def __init__(self, config):
        self.config = config

        # if self.config.overwrite:
        #     data_root = self.config.data_root
        #     shutil.rmtree(os.path.join(data_root, self.config.repo_id), ignore_errors=True)
        
        self.create_dataset()

    def get_today_time(self):
        # 获取当前日期和时间
        today = datetime.datetime.now()
        
        # 格式化日期为字符串，格式为 "YYYY-MM-DD"
        date_string = today.strftime("%Y%m%d%H%M%S")
        return date_string
    
    def create_dataset(self):
        # 构建 features 字典
        features = {}
        data_root = os.path.join(self.config.data_root, self.get_today_time())
        print(data_root)
        for camera_dict in self.config.rgb_names:
            rgb_name = next(iter(camera_dict))  # 获取字典的键（如 'cam_high'）
            features[rgb_name] = camera_dict[rgb_name]  # 赋值配置
        
        # rgb_config = {
        #     'dtype': 'video',
        #     'shape': (self.config.image_height, self.config.image_width, 3),
        #     'name': ['height', 'width', 'channels'],
        # }
        # features = {rgb_name: self.config.rgb_names[rgb_name] for rgb_name in self.config.rgb_names}
        features['states'] = {
            'dtype': 'float32',
            'shape': (self.config.action_len,),
            'name': self.config.action_name,
        }
        features['actions'] = {
            'dtype': 'float32',
            'shape': (self.config.action_len,),
            'name': self.config.action_name,
        }

        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            root=data_root,
            robot_type = self.config.robot,
            video_backend = self.config.video_backend,
            features=features,
        )
    
    def process_data(self):
        for source_idx, source_data_root in enumerate(self.config.source_data_roots):
            episode_path = source_data_root
            print(f'Processing source {source_idx + 1}, {episode_path}')
            #self._load_episode2(episode_path)
            self._add_episode2(episode_path)
    
    # def _add_episode(self, episode_path):
    #     raw_images, raw_actions, instruction = self._load_episode(episode_path)
    #     indexs = list(range(len(raw_images[self.config.rgb_names[0]])))
        
    #     for i in tqdm(indexs[:-1], desc=f'Adding episode {episode_path}'):
    #         states = np.concatenate([raw_actions[action_dir][i] for action_dir in self.config.action_dirs])
    #         actions = np.concatenate([raw_actions[action_dir][i + 1] for action_dir in self.config.action_dirs])
    #         if not self._check_nonoop_actions(states, actions):
    #             continue

    #         frame = {rgb_name: load_image(raw_images[rgb_name][i]) for rgb_name in self.config.rgb_names}
    #         frame['states'] = states
    #         frame['actions'] = actions
    #         self.dataset.add_frame(frame, task=instruction)
            
    #     self.dataset.save_episode()
        
    # def _load_episode(self, episode_path):
    #     raw_images = defaultdict(list)
    #     for rgb_dir, rgb_name in zip(self.config.rgb_dirs, self.config.rgb_names):
    #         rgb_dir = os.path.join(episode_path, rgb_dir)
    #         for file_name in sorted(os.listdir(rgb_dir), key=lambda x: float(x[:-5])):
    #             image_path = os.path.join(rgb_dir, file_name)
    #             raw_images[rgb_name].append(image_path)
            
    #     raw_actions = defaultdict(list)
    #     for action_dir, action_keys in zip(self.config.action_dirs, self.config.action_keys_list):
    #         action_dir_ = os.path.join(episode_path, action_dir)
    #         for file_name in sorted(os.listdir(action_dir_), key=lambda x: float(x[:-5])):
    #             action_path = os.path.join(action_dir_, file_name)
    #             with open(action_path, 'r') as f:
    #                 action_data = json.load(f)
    #             action_data = np.array([action_data[key] for key in action_keys])
    #             raw_actions[action_dir].append(action_data)
        
    #     instruction_path = os.path.join(episode_path, self.config.instruction_path)
    #     with open(instruction_path, 'r') as f:
    #         instruction_data = json.load(f)
    #     instruction = instruction_data['instructions'][0]
    #     if instruction == 'null':
    #         instruction = self.config.default_instruction
        
    #     lens = []
    #     for rgb_name, images_list in raw_images.items():
    #         lens.append(len(images_list))
    #     for action_dir, actions_list in raw_actions.items():
    #         lens.append(len(actions_list))
        
    #     assert all(lens[0] == l for l in lens), "All lists must have the same length"
        
    #     return raw_images, raw_actions, instruction
    
    def _add_episode2(self, episode_path):
        index, raw_images, raw_actions, instruction = self._load_episode2(episode_path)
        #indexs = list(range(len(raw_images[self.config.rgb_names[0]])))
        print(index)

        for i in tqdm(range(index), desc=f'Adding episode {episode_path}'):
            #states = np.concatenate([raw_actions[action_dir][i] for action_dir in self.config.action_dirs])
            #actions = np.concatenate([raw_actions[action_dir][i + 1] for action_dir in self.config.action_dirs])
            states = raw_actions[i,:]
            actions = raw_actions[i,:]
            # if not self._check_nonoop_actions(states, actions):
            #     continue

            frame = {next(iter(rgb_name)): cv2.imdecode(np.frombuffer(raw_images[next(iter(rgb_name))][i], dtype=np.uint8), cv2.IMREAD_COLOR) for rgb_name in self.config.rgb_names}
            frame['states'] = states
            frame['actions'] = actions
            frame['task'] = instruction
            self.dataset.add_frame(frame)
            
        self.dataset.save_episode()
    

    def _load_episode2(self, episode_path):
        try:
            with h5py.File(episode_path, "r") as file:
                print("Datasets in HDF5 file:")
                
                def print_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"Dataset: {name}")
                        print(f"  Shape: {obj.shape}, DType: {obj.dtype}")
                        if obj.attrs:
                            print("  Attributes:")
                            for key, val in obj.attrs.items():
                                print(f"    {key}: {val}")
                
                #file.visititems(print_datasets)  # 打印 dataset 信息
                
                # 存储所有 dataset 的字典（使用 name 作为键）
                datasets_dict = {}
                
                def load_datasets(group, path=''):
                    for key in group.keys():
                        current_path = f"{path}/{key}" if path else key
                        item = group[key]
                        
                        if isinstance(item, h5py.Dataset):
                            # 直接使用 current_path（即 name）作为字典键
                            datasets_dict[current_path] = item[()]  # 读取数据
                            print(f"Loaded dataset: {current_path}")
                            
                        elif isinstance(item, h5py.Group):
                            load_datasets(item, current_path)  # 递归处理子组

                raw_images = defaultdict(list)
                load_datasets(file)  # 加载所有 dataset
                for dataset_path in datasets_dict:
                    if "action" in dataset_path:
                        frame = datasets_dict[dataset_path].shape
                        raw_actions = datasets_dict[dataset_path][:,self.config.action_index] 
                    if "cam_high" in dataset_path:
                        raw_images["cam_high"] = datasets_dict[dataset_path]
                    if "cam_left_wrist" in dataset_path:
                        raw_images["cam_left_wrist"] = datasets_dict[dataset_path]
                    if "cam_right_wrist" in dataset_path:
                        raw_images["cam_right_wrist"] = datasets_dict[dataset_path]
                        # raw_actions = defaultdict(list)
                        # for action_dir, action_keys in zip(self.config.action_dirs, self.config.action_keys_list):
                        #     action_dir_ = os.path.join(episode_path, action_dir)
                        #     print(action_dir)
                        # #这是action的数据，根据查看里面的信息是Dataset Shape: (1738, 128) Dataset dtype: float32

                instruction = self.config.default_instruction
                return frame[0],raw_images,raw_actions,instruction
                
        except Exception as e:
            print(f"Error loading episode: {str(e)}")
            return None
        
    def _check_nonoop_actions(self, states, actions):
        return np.abs(states - actions).max() > self.config.nonoop_threshold


def main():
    config = DataConfig()
    processor = DataProcessor(config)
    processor.process_data()


if __name__ == "__main__":
    main()

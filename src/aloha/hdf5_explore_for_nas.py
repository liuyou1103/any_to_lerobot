import h5py
import datetime
import time
import pandas as pd
from nas_sdk import NASAuthenticator
from aloha_config import DataConfig
import os
import json

def read_csv(path):
    df = pd.read_csv(path)
    # 检查是否存在 'trans' 列
    if 'trans' in df.columns:
        # 过滤掉 'trans' 列为 'Y' 的行
        filtered_df = df[~df['trans'].isin(['Y', 'N'])]
    else:
        # 如果没有 'trans' 列，则使用全部数据
        filtered_df = df
    # 提取 'json_path' 和 'nas_path' 两列
    selected_columns = filtered_df[['json_path', 'nas_path']]
    # 转换成嵌套列表（每行是一个子列表）
    nested_list = selected_columns.values.tolist()
    # 打印前3行示例（可选）
    # print("嵌套列表（前3行）:", nested_list[:3])
    return nested_list

def read_csv_strange(path):
    df = pd.read_csv(path)
    # 检查是否存在 'trans' 列
    # if 'trans' in df.columns:
    #     # 过滤掉 'trans' 列为 'Y' 的行
    #     filtered_df = df[~df['trans'].isin(['Y', 'N'])]
    # else:
    #     # 如果没有 'trans' 列，则使用全部数据
    #     filtered_df = df
    filtered_df = df
    # 提取 'json_path' 和 'nas_path' 两列
    selected_columns = filtered_df[['原始数据路径', 'json文件路径']]
    # 转换成嵌套列表（每行是一个子列表）
    nested_list = selected_columns.values.tolist()
    # 打印前3行示例（可选）
    # print("嵌套列表（前3行）:", nested_list[:3])
    return nested_list

def download_from_nas(nas_path, an_path):
        # 定义文件夹名称映射关系（中文键 -> 英文值）
        
        # parts = []
        # last_name = ""            
        # # 解析an_path路径（最多循环10次防止死循环）
        # for _ in range(10):
        #     path, tail = os.path.split(an_path)
        #     print(path)
        #     if not tail:  # 路径解析完毕
        #         break                   
        #     # 如果tail是映射键，使用映射值；否则保持原样
        #     processed_tail = fold_name.get(tail, tail)                
        #     if 'aloha' in processed_tail:  # 遇到realman终止解析
        #         break   
        #     if 'raw_data' in processed_tail:  # 遇到realman终止解析
        #         break                 
        #     parts.append(processed_tail)
        #     an_path = path           
        # # 反向构建相对路径（使用os.path.join保证跨平台兼容性）
        # if parts:
        #     last_name = os.path.join(*reversed(parts))          
        # # 构建本地和NAS的完整路径
        # print(last_name)
        # local_path_dir = os.path.basename(nas_path.rstrip('/'))  # 去除末尾可能的斜杠
        # nas_path = nas_path + '.hdf5'
        # 根据不同前缀处理
        nas_path = nas_path.rstrip()
        if nas_path.startswith('/volume1'):
            # 如果是 'volume' 开头的相对路径，去掉 'volume' 前缀并拼接 .hdf5
            print(nas_path)
            nas_path = os.path.relpath(nas_path, '/volume1')
            if not nas_path.endswith('.hdf5'):
                nas_path = '/' + nas_path + '.hdf5'
            print(nas_path)
        if nas_path.startswith('/'):
            # 如果是绝对路径，直接拼接 .hdf5（确保不重复）
            if not nas_path.endswith('.hdf5'):
                nas_path += '.hdf5'
        else:
            # 其他情况，当作相对路径处理
            nas_path = '/' + nas_path
            if not nas_path.endswith('.hdf5'):
                nas_path += '.hdf5'

        relative_nas_path = os.path.relpath(nas_path, '/')  # 去掉开头的 '/'，变成相对路径
        local_path = os.path.join(config.hdf5_root, relative_nas_path)
        # nas_path = os.path.dirname(nas_path)
        # nas_path = os.path.join(nas_path, last_name)            
        # 执行下载并返回结果
        print(nas_path,local_path)
        if nas_auth.download_file(nas_path, local_path):
            return nas_path, local_path
        return None, None




config = DataConfig()
nas_auth = NASAuthenticator()     
if not nas_auth.get_auth_sid():
    print("no nas")
else:
    nested_list = read_csv(config.source_data_csv)
    read_csv_strange
    raw_data_dir = os.path.dirname(config.source_data_csv)
    for both_data in nested_list:
        print("new task")
        json_path = os.path.join(raw_data_dir,both_data[0])
        nas_path = both_data[1]
        task_name = os.path.basename(nas_path)
        task_name_strange = task_name + '.csv'
        csv_strange_path = os.path.join(config.source_data_csv_path,task_name_strange)
        nested_list_strange = read_csv_strange(csv_strange_path)
        print(task_name)
        if os.path.exists(json_path):
            with open(json_path,"r",encoding='utf-8') as file:
                data = json.load(file)
            for source_idx,hdf5_file in enumerate(data):
                episode_path = hdf5_file["path"]
                print(episode_path)
                for item in nested_list_strange:
                    
                    if episode_path in str(item[1]):
                        nas_path_1 = item[0]
                        break
                nas_path, local_path = download_from_nas(nas_path_1,episode_path)
                print(local_path)
                # with h5py.File(local_path, "r") as file:
                #     print("Datasets in HDF5 file:")
                #     def info_datasets(name, obj):
                #         if isinstance(obj, h5py.Dataset):
                #             print(f"Dataset: name")
                #             print(f"  Shape: obj.shape, DType: obj.dtype")
                #             if obj.attrs:
                #                 print("  Attributes:")
                #                 for key, val in obj.attrs.items():
                #                     print(f"    key: val")
        
                # info_datasets(file)  # 打印 dataset 信息
                with h5py.File(local_path, "r") as file:
                    def list_hdf5_structure(hdf5_file, indent=0):
                        """递归列出HDF5文件的结构"""
                        for key in hdf5_file.keys():
                            print(' ' * indent + f"- {key} ({'Group' if isinstance(hdf5_file[key], h5py.Group) else 'Dataset'})")
                            if isinstance(hdf5_file[key], h5py.Group):
                                list_hdf5_structure(hdf5_file[key], indent + 2)
                    list_hdf5_structure(file)
                break
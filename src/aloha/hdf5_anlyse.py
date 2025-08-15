import h5py
import numpy as np

def export_hdf5_to_txt(hdf5_file_path, output_dir):
    # 打开HDF5文件
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # 遍历HDF5文件中的所有组和数据集
        def visit_groups(name, obj):
            if isinstance(obj, h5py.Dataset):
                # 获取数据集的数据和精度
                data = obj[:]
                dtype = obj.dtype
                
                # 生成输出文件路径
                output_file_path = f"{output_dir}/{name.replace('/', '_')}.txt"
                
                # 将数据写入TXT文件
                np.savetxt(output_file_path, data.flatten() if data.ndim > 1 else data, fmt='%s')
                
                # 输出数据精度信息
                print(f"Exported {name} to {output_file_path} with dtype: {dtype}")
        
        # 访问HDF5文件中的所有对象
        hdf5_file.visititems(visit_groups)

# 使用示例
hdf5_file_path = "your_file.h5"  # 替换为你的HDF5文件路径
output_dir = "output_txt_files"  # 替换为你想要保存TXT文件的目录

# 确保输出目录存在
import os
os.makedirs(output_dir, exist_ok=True)

# 执行导出
export_hdf5_to_txt('/home/diy01/aloha/aloha/docker/1.上传数据/WD14T/松灵分体/task_Basket_banana_500_5.13/task_black_left_Basket_banana_125_5.12/episode_110.hdf5', '/home/diy01/lerobot/src/aloha')
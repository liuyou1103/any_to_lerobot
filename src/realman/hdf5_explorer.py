import h5py
import argparse

def list_hdf5_structure(hdf5_file, indent=0):
    """递归列出HDF5文件的结构"""
    for key in hdf5_file.keys():
        print(' ' * indent + f"- {key} ({'Group' if isinstance(hdf5_file[key], h5py.Group) else 'Dataset'})")
        if isinstance(hdf5_file[key], h5py.Group):
            list_hdf5_structure(hdf5_file[key], indent + 2)

def export_dataset_to_txt(dataset, output_file):
    """将数据集导出到TXT文件"""
    with open(output_file, 'w') as f:
        # 写入数据集形状信息
        f.write(f"Dataset Shape: {dataset.shape}\n")
        f.write(f"Dataset dtype: {dataset.dtype}\n")
        f.write("-" * 50 + "\n")
        
        # 将数据转换为numpy数组并展平为一维（简化输出）
        data = dataset[()]
        if data.ndim > 1:
            # 对于多维数据，逐行写入
            for i, row in enumerate(data):
                f.write(f"Row {i}: {row}\n")
        else:
            # 一维数据直接写入
            f.write(str(data))

def process_hdf5_file(input_file, output_prefix):
    """处理HDF5文件：列出结构并导出数据集"""
    with h5py.File(input_file, 'r') as f:
        print("\nHDF5 File Structure:")
        print("====================")
        list_hdf5_structure(f)
        
        print("\nExporting datasets to TXT files...")
        
        def save_datasets(group, path=''):
            for key in group.keys():
                current_path = f"{path}/{key}" if path else key
                item = group[key]
                
                if isinstance(item, h5py.Dataset):
                    # 为每个数据集创建单独的输出文件
                    output_file = f"{output_prefix}_{current_path.replace('/', '_')}.txt"
                    print(f"Exporting {current_path} -> {output_file}")
                    export_dataset_to_txt(item, output_file)
                elif isinstance(item, h5py.Group):
                    save_datasets(item, current_path)
        
        save_datasets(f)

def dd():
    with h5py.File("../aloha/episode_1.hdf5", "r") as f:
        print("Action shape:", f["action"].shape)  # 预期 (N, 7)，实际 (N, 128)
        print("State shape:", f["observation/state"].shape)  # 预期 (N, 15)，实际 (N, 128)
        print("Image shape:", f["observation/images/camera_high"].shape)  # 预期 (N, H, W, 3)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='HDF5 File Explorer and Exporter')
    # parser.add_argument('input_file', help='Path to the input HDF5 file')
    # parser.add_argument('-o', '--output', default='output', 
    #                     help='Prefix for output TXT files (default: "output")')
    # args = parser.parse_args()
    
    process_hdf5_file('/home/ctos/raw_data/realman/task_stack_basket_500_4.27/brown_down/episode_76.hdf5', './')
    print("\nDone!")
    # dd()
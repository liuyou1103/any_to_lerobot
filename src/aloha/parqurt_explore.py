
import pandas as pd
import numpy as np

# 读取 Parquet 文件
file_path = '/home/diy01/lerobot/aloha/20250728181258/aloha_task_pull_plug_200_4.29/data/chunk-000/episode_000000.parquet'
df = pd.read_parquet(file_path)

# 获取第一行数据
first_row = df.iloc[0]

# 定义函数：获取数据的形状（支持标量和数组）
def get_shape(value):
    if isinstance(value, (list, np.ndarray, pd.Series)):
        return len(value)  # 数组/列表的长度
    else:
        return "scalar"    # 标量值

# 收集形状信息
shape_info = {
    "columns": list(df.columns),
    "num_columns": len(df.columns),
    "shapes": {col: get_shape(first_row[col]) for col in df.columns},
    "dtypes": df.dtypes.to_dict()
}

# 将第一行数据和形状信息写入文本文件
output_file = 'first_row_with_detailed_shape.txt'
with open(output_file, 'w') as f:
    f.write("=== 第一行数据 ===\n")
    f.write(first_row.to_string() + "\n\n")
    
    f.write("=== 形状信息 ===\n")
    f.write(f"总列数: {shape_info['num_columns']}\n")
    f.write("列名及形状:\n")
    for col in shape_info['columns']:
        shape = shape_info['shapes'][col]
        dtype = shape_info['dtypes'][col]
        f.write(f"  {col}: shape={shape}, dtype={dtype}\n")

print(f"第一行数据和详细形状信息已写入 {output_file}")
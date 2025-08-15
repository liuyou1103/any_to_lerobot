import pyarrow.parquet as pq

def inspect_parquet_with_pyarrow(file_path, output_txt=None):
    # 读取 Parquet 文件
    parquet_file = pq.ParquetFile(file_path)
    
    # 1. 打印 Schema（列名和数据类型）
    print("Schema (列名和数据类型):")
    columns_info = []
    for field in parquet_file.schema:
        col_info = f"{field.name}: {field.physical_type}"
        print(f"  {col_info}")
        columns_info.append(col_info)
    
    # 2. 打印总行数
    total_rows = parquet_file.metadata.num_rows
    print(f"\n总行数: {total_rows}")
    
    # 3. 打印行组信息
    print("行组信息:")
    row_group_info = []
    for i in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(i)
        info = f"行组 {i}: {row_group.num_rows} 行"
        print(f"  {info}")
        row_group_info.append(info)
    
    # 4. 读取前5行数据（小文件适用）
    print("\n前5行数据:")
    table = parquet_file.read()
    df = table.to_pandas()
    print(df.head())
    
    # 5. 输出到 txt 文件（如果指定了 output_txt）
    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            # 写入列名
            f.write("=== 列名和数据类型 ===\n")
            f.write("\n".join(columns_info))
            f.write("\n\n")
            
            # 写入总行数
            f.write(f"总行数: {total_rows}\n\n")
            
            # 写入行组信息
            f.write("=== 行组信息 ===\n")
            f.write("\n".join(row_group_info))
            f.write("\n\n")
            
            # 写入第一行数据
            f.write("=== 第一行数据 ===\n")
            if total_rows > 0:
                first_row = df.iloc[0].to_dict()  # 转为字典格式
                for key, value in first_row.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write("(空文件，无数据)\n")
        
        print(f"\n信息已保存到: {output_txt}")

# 调用示例
inspect_parquet_with_pyarrow(
    file_path="/home/ctos/pika/convert/task_pour_basket_egg_yolk_pastry_1000_5.11/data/chunk-000/episode_000000.parquet",
    output_txt="parquet_info_output1_4.txt"
)
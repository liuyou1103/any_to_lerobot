from nas_sdk import NASAuthenticator
import pandas as pd
import os


from nas_sdk import NASAuthenticator

import os

 

def upload_folder(local_folder, nas_folder):
    nas_auth = NASAuthenticator()
    if not nas_auth.get_auth_sid():
        print("NAS 认证失败，无法上传")
        return

    # 确保 NAS 路径以斜杠结尾
    if not nas_folder.endswith('/'):
        nas_folder += '/'
 
    # 获取本地文件夹的名称（最后一级目录）
    local_folder_name = os.path.basename(local_folder.rstrip('/')) 
    # 最终 NAS 目标路径 = nas_folder + local_folder_name
    nas_target_folder = os.path.join(nas_folder, local_folder_name)
    # 遍历本地文件夹
    for root, dirs, files in os.walk(local_folder):
        # 计算相对路径（相对于 local_folder）
        relative_path = os.path.relpath(root, local_folder)
        # 构造 NAS 目标路径（保持相同的子目录结构）
        if relative_path == '.':
            nas_target = nas_target_folder  # 根目录
        else:
            nas_target = os.path.join(nas_target_folder, relative_path)

        

        # 上传每个文件
        for file in files:
            local_file_path = os.path.join(root, file)
            nas_file_path = os.path.join(nas_target, file)
            print(f"上传 {local_file_path} 到 {nas_file_path}")
            nas_auth.upload(local_file_path, nas_file_path)

 

if __name__ == "__main__":
    # 配置路径
    nas_path = '/docker/2.lerobot/realman'  # NAS 基础路径
    local_parent_path = '/home/ctos/realman_wait_for_upload'
    local_path_list = [
        'realman_task_Basket _orang_500_5.11',
        'realman_task_basket_store_egg_tart_500_5.06',
        'realman_task_document_realman_500_4.24',
        'realman_task_fold_towels_500_5.05'
    ]

    for local_path in local_path_list:
        local_path = os.path.join(local_parent_path,local_path)
        if not os.path.exists(local_path):
            print(f"错误：本地路径 {local_path} 不存在")
            exit(1)

        print(f"开始上传 {local_path} 到 {nas_path}")
        upload_folder(local_path, nas_path)
        print("上传完成")
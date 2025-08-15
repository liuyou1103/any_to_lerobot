import numpy as np

def recover_rotation_matrix(col1, col2):
    """
    从旋转矩阵的前两列恢复完整的3x3旋转矩阵
    
    参数:
    col1, col2 (array-like): 旋转矩阵的前两列，形状为(3,)
    
    返回:
    numpy.ndarray: 3x3的旋转矩阵
    """
    col1 = np.array(col1, dtype=np.float32)
    col2 = np.array(col2, dtype=np.float32)
    
    # 检查向量是否接近零
    eps = 1e-10  # 设定一个小的阈值
    
    # 确保col1是单位向量
    norm_col1 = np.linalg.norm(col1)
    if norm_col1 < eps:
        raise ValueError("col1 几乎为零向量，无法归一化")
    col1 = col1 / norm_col1
    
    # 从col2中移除col1的投影，使其正交
    col2 = col2 - np.dot(col2, col1) * col1
    
    # 确保col2也是单位向量
    norm_col2 = np.linalg.norm(col2)
    if norm_col2 < eps:
        # 如果正交后的col2几乎为零，尝试寻找与col1正交的向量
        # 这里选择一个与col1正交的任意向量
        if abs(col1[0]) < 0.5:
            temp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            temp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        col2 = temp - np.dot(temp, col1) * col1
        col2 = col2 / np.linalg.norm(col2)
    else:
        col2 = col2 / norm_col2
    
    # 计算第三列，已经是单位向量
    col3 = np.cross(col1, col2)
    
    # 组合成旋转矩阵
    R = np.column_stack((col1, col2, col3))
    
    # 检查行列式是否为1（确保是旋转而非反射）
    if np.linalg.det(R) < 0:
        # 如果行列式为负，翻转第三列
        R[:, 2] = -R[:, 2]
    
    return R

def rotation_matrix_to_quaternion(R):
    """
    将3x3旋转矩阵转换为四元数表示
    
    参数:
    R (numpy.ndarray): 3x3的旋转矩阵
    
    返回:
    numpy.ndarray: 四元数，格式为[w, x, y, z]
    """
    R = np.array(R, dtype=np.float32)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    # 归一化四元数，提高数值稳定性
    quaternion = np.array([w, x, y, z], dtype=np.float32)
    return quaternion / np.linalg.norm(quaternion)

def test_rotation_recovery():
    """测试旋转矩阵恢复和四元数转换功能"""
    # 示例：假设已知旋转矩阵的前两列
    col1_example = [1.6786905e-02, 6.0468137e-02, -9.9802893e-01]
    col2_example = [-9.4887757e-01, -3.1370163e-01, -3.4966592e-02]
    
    try:
        R_recovered = recover_rotation_matrix(col1_example, col2_example)
        quaternion = rotation_matrix_to_quaternion(R_recovered)
        
        print("恢复的旋转矩阵:")
        print(R_recovered)
        print("\n四元数:", quaternion)
        
        # 验证旋转矩阵的性质
        print("\n验证旋转矩阵性质:")
        print("行列式:", np.linalg.det(R_recovered))
        print("R^T * R 接近单位矩阵:", np.allclose(R_recovered.T @ R_recovered, np.eye(3)))
        
        # 验证四元数模长
        print("\n四元数模长:", np.linalg.norm(quaternion))
        
    except ValueError as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_rotation_recovery()
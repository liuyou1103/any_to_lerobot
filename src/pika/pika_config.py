import math
from dataclasses import dataclass


@dataclass
class DataConfig:
    overwrite = True
    check_only = False

    source_data_roots = [
        # 'examples/pika_example_data',
        '/home/shihanwu/Datasets/pika-demo',
    ]

    image_height = 480
    image_width = 640
    #RGB 图像所在的子目录路径
    rgb_dirs = [
        'camera/color/camera_realsense_c',
        'camera/color/pikaDepthCamera_l',
        'camera/color/pikaFisheyeCamera_l',
        'camera/color/pikaDepthCamera_r',
        'camera/color/pikaFisheyeCamera_r',
    ]
    #RGB 图像的名称
    rgb_names = [
        'observation.images.cam_center',
        'observation.images.cam_left_wrist',
        'observation.images.cam_left_wrist_fisheye',
        'observation.images.cam_right_wrist',
        'observation.images.cam_right_wrist_fisheye',
    ]

    use_depth = True
    depth_dirs = [
        'camera/depth/pikaDepthCamera_c',
        'camera/depth/pikaDepthCamera_l',
        'camera/depth/pikaDepthCamera_r',
    ]
    depth_names = [
        'observation.depths.cam_center',
        'observation.depths.cam_left_wrist',
        'observation.depths.cam_right_wrist',
    ]

    action_name = 'action'
    action_dirs = [
        'localization/pose/pika_l',
        'gripper/encoder/pika_l',
        'localization/pose/pika_r',
        'gripper/encoder/pika_r',
    ]
    action_keys_list = [
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
        ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
        ['angle'],
    ]
    action_keys_list_2 = [
                ["left_end_positions_x",
                "left_end_positions_y",
                "left_end_positions_z",
                "left_end_euler_roll",
                "left_end_euler_pitch",
                "left_end_euler_yaw"],
                ["left_gripper_joint"],
                ["right_end_positions_x",
                "right_end_positions_y",
                "right_end_positions_z",
                "right_end_euler_roll",
                "right_end_euler_pitch",
                "right_end_euler_yaw"],
               ["right_gripper_joint"],
            ]
    
    #位置无操作阈值，用于判断位置变化是否足够小以视为无操作
    position_nonoop_threshold = 1e-3 # 0.1cm
    #旋转无操作阈值，用于判断旋转变化是否足够小以视为无操作
    rotation_nonoop_threshold = math.pi / 180 # 1 degree
    use_delta = True

    use_state = True
    state_name = 'observation.state'

    instruction_path = 'instructions.json'
    default_instruction = 'do something'

    repo_id = 'Koorye/pika-demo'
    data_root = None
    fps = 30
    video_backend = 'pyav'

#设备信息，device_info.json
    device_info = {
        "device_list": [
        {
            "device_id": "9xK992pQ",
            "device_type" : "Pika",
            "device_type_info":"Pika Sense",
            "device_info": {           
                "cam_center": {
                    "type": "Intel RealSense D435"
                },
                "cam_right_wrist_fisheye":{
                    "type":"DECXIN CAMERA"
                },
                "cam_right_wrist":{
                    "type":"Intel RealSense D405"
                },
                "cam_left_wrist_fisheye":{
                    "type":"DECXIN CAMERA"
                },  
                "cam_left_wrist":{
                    "type":"Intel RealSense D405"
                },
                "gripper_left":{
                    "type":None,
                    "dimension":{            
                            'left_end_positions_x': 'm',
                            'left_end_positions_y': 'm',
                            'left_end_positions_z': 'm',
                            'left_end_euler_roll': 'rad',
                            'left_end_euler_pitch': 'rad',
                            'left_end_euler_yaw': 'rad',
                            'left_gripper_joint': 'rad',
                    },
                    "description":{
                            
                            'left_end_positions_x': None,
                            'left_end_positions_y': None,
                            'left_end_positions_z': None,
                            'left_end_euler_roll': None,
                            'left_end_uler_pitch': None,
                            'left_end_euler_yaw': None,
                            'left_gripper_joint': None,
                    }
                },
                "gripper_right":{
                    "type":None,
                    "dimension":{
                            'right_end_positions_x': 'm',
                            'right_end_positions_y': 'm',
                            'right_end_positions_z': 'm',
                            'right_end_euler_roll': 'rad',
                            'right_end_euler_pitch': 'rad',
                            'right_end_euler_yaw': 'rad',
                            'right_gripper_joint': 'rad',
                    },
                    "description":{
                            'right_end_positions_x': None,
                            'right_end_positions_y': None,
                            'right_end_positions_z': None,
                            'right_end_euler_roll': None,
                            'right_end_euler_pitch': None,
                            'right_end_euler_yaw': None,
                            'right_gripper_joint': None,
                    }
                }

            }
        }
    ]
    }

#摄像头信息，camera_intrinsic.json
    camera_info = {           
    "device_list": [
        {
            "device_id": "9xK992pQ",
            "calibration_info": {    
                "version": "1.0",   
                "date": "2025-07-17",
                "source": "calibration_tool", 
                "reprojection_error": None   
            },
            "camera_intrinsics": {    
                "center_RGB": {
                    "model": "PINHOLE",
                    "parameters": {
                        "width": 640,
                        "height": 480,
                        "fx": 604.316,
                        "fy": 603.979,
                        "cx": 328.597,
                        "cy": 254.841
                    },
                    "distortion": {      
                        "model": "Inverse Brown Conrady",
                        "k1": 0,
                        "k2": 0,
                        "k3": 0,
                        "p1": 0,
                        "p2": 0
                    }
                },
                "center_Depth": {
                    "model": "PINHOLE",
                    "parameters": {
                        "width": 640,
                        "height": 480,
                        "fx": 386.353,
                        "fy": 386.353,
                        "cx": 319.706,
                        "cy": 246.779
                    },
                    "distortion": {      
                        "model": "Brown Conrady",
                        "k1": 0,
                        "k2": 0,
                        "k3": 0,
                        "p1": 0,
                        "p2": 0
                    }
                },
                "right_RGB": {
                    "model": "PINHOLE",
                    "parameters": {
                        "width": 640,
                        "height": 480,
                        "fx": 385.208,
                        "fy": 385.208,
                        "cx": 321.609,
                        "cy": 241.442
                    },
                    "distortion": {
                        "model": "Brown Conrady",
                        "k1": 0,
                        "k2": 0,
                        "k3": 0,
                        "p1": 0,
                        "p2": 0
                    }
                },
                "right_Depth": {
                    "model": "PINHOLE",
                    "parameters": {
                        "width": 640,
                        "height": 480,
                        "fx": 385.208,
                        "fy": 385.208,
                        "cx": 321.609,
                        "cy": 241.442
                    },
                    "distortion": {
                        "model": "Brown Conrady",
                        "k1": 0,
                        "k2": 0,
                        "k3": 0,
                        "p1": 0,
                        "p2": 0
                    }
                },
                "left_RGB": {
                    "model": "PINHOLE",
                    "parameters": {
                        "width": 640,
                        "height": 480,
                        "fx": 435.195,
                        "fy": 434.46,
                        "cx": 317.356,
                        "cy": 241.957
                    },
                    "distortion": {
                        "model": "Inverse Brown Conrady",
                        "k1": -0.0567643,
                        "k2": 0.0618577,
                        "k3": -0.0210108,
                        "p1": -0.000129418,
                        "p2": 0.00168662
                    }
                },               
                "left_Depth": {
                    "model": "PINHOLE",
                    "parameters": {
                        "width": 640,
                        "height": 480,
                        "fx": 383.883,
                        "fy": 383.883,
                        "cx": 313.602,
                        "cy": 239.62
                    },
                    "distortion": {
                        "model": "Brown Conrady",
                        "k1": 0,
                        "k2": 0,
                        "k3": 0,
                        "p1": 0,
                        "p2": 0
                    }
                }     
            }
        }
    ]
    }

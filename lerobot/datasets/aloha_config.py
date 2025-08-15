from dataclasses import dataclass


@dataclass
class DataConfig:
    overwrite = True

    source_data_roots = [
        '/home/liuyou/Documents/dataset/aloha/episode_1.hdf5',
        '/home/liuyou/Documents/dataset/aloha/episode_2.hdf5',
    ]

    # image_height = 480
    # image_width = 640
    # rgb_dirs = [
    #     'camera/color/camera_intel_c',
    #     'camera/color/camera_intel_l',
    #     'camera/color/camera_intel_r',
      
    # ]
    rgb_names = [
        {'cam_high':{
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": "false",
                "has_audio": "false"
            }
        }},
        {'cam_left_wrist':{
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": "false",
                "has_audio": "false"
            }
        }},
        {'cam_right_wrist':{
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.fps": 25.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": "false",
                "has_audio": "false"
            }
        }},
    ]

    action_len = 32
    # action_dirs = [
    #     'localization/pose/aloha_l',
    #     'gripper/encoder/aloha_l',
    #     'localization/pose/aloha_r',
    #     'gripper/encoder/aloha_r',
    # ]
    # action_keys_list = [
    #     ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
    #     ['angle'],
    #     ['x', 'y', 'z', 'roll', 'pitch', 'yaw'],
    #     ['angle'],
    # ]
    action_name = [
        'Right arm joint positions(rad,rad,rad,rad,rad,rad)',
        'Right gripper joint positions(rad)',
        'Right end effector positions(m,m,m)',
        'Right end effector 6D pose(6)',
        'Left arm joint positions(rad,rad,rad,rad,rad,rad)',
        'Left gripper joint positions(rad)',
        'Left end effector positions(m,m,m)',
        'Left end effector 6D pose(6)'
    ]
    action_index = [0,1,2,3,4,5,10,30,31,32,
                    33,34,35,36,37,38,50,51,
                    52,53,54,55,60,80,81,82,
                    83,84,85,86,87,88]
    nonoop_threshold = 1e-3

    instruction_path = 'instructions.json'
    default_instruction = 'Fold clothes'

    repo_id = 'fold_clothes'
    data_root = '/home/liuyou/lerobot/aloha/'
    fps=25
    video_backend = 'pyav'
    robot = 'aloha'



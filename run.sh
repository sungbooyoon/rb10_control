# 1. Teleoperation
# 1.1 setup robot and camera
ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false
ros2 launch realsense2_camera rs_launch.py 

# 1.2 teleoperation
python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_teleop.py



# 2. Robot learning
# 2.1 setup camera
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true

# 2.2 collect dataset
python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_demo_recorder.py

# 2.3 playback one demo
ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false
python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_demo_playback.py



# Others
## Apritag
ros2 run apriltag_ros apriltag_node --ros-args \ 
    -r image_rect:=/camera/camera/color/image_raw \
    -r camera_info:=/camera/camera/color/camera_info \
    --params-file /home/sungboo/ros2_ws/src/apriltag/apriltag_ros/cfg/tags_36h11.yaml

## handeye calibration
ros2 launch easy_handeye2 handeye_calibrate.launch.py

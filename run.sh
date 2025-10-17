# 1. Teleoperation
# 1.1 setup robot and camera
ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false
ros2 launch realsense2_camera rs_launch.py 

# 1.2 teleoperation
python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_teleop.py



# 2. Robot learning
# 2.1 setup camera
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true
ros2 run tf2_ros static_transform_publisher \
  -0.062 -0.005 0  0.5 -0.5 -0.5 0.5  tcp camera_link

# 2.2 collect dataset
# 2.2.1 kinesthetic teaching mode
python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_controller.py

python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_demo_recorder_bridge.py

cd /home/sungboo/ros2_ws/src/rb10_control/dataset
ros2 bag record /tf /rb/joint_states /rb/tcp_pose /rb/freedrive /rb/ee_wrench /camera/camera/color/image_raw

python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_rosbag_to_hdf5.py \
  --folder /home/sungboo/ros2_ws/src/rb10_control/dataset/251017_kin_3 \
  --out /home/sungboo/ros2_ws/src/rb10_control/dataset/251017_kin_3.hdf5 \
  --no-normalize-actions --freedrive-only

# 2.2.2 teleoperation mode
python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_controller.py
python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_teleop.py

TODO: demo recorder - teleop input as actions

python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_rosbag_to_hdf5.py \
  --folder /home/sungboo/ros2_ws/src/rb10_control/dataset/251017_tel_1 \
  --out /home/sungboo/ros2_ws/src/rb10_control/dataset/251017_tel_1.hdf5 \
  --no-normalize-actions --no-freedrive-only

# 2.3 playback one demo
ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false
python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_demo_playback.py



# Others
## Apritag
ros2 run apriltag_ros apriltag_node --ros-args \
    -r image_rect:=/camera/camera/color/image_raw \
    -r camera_info:=/camera/camera/color/camera_info \
    --params-file /home/sungboo/ros2_ws/src/apriltag_ros/cfg/tags_36h11.yaml

## handeye calibration
ros2 launch easy_handeye2 handeye_calibrate.launch.py \
  name:=my_eih_calib \
  calibration_type:=eye_in_hand \
  robot_base_frame:=link0 \
  robot_effector_frame:=tcp \
  tracking_base_frame:=camera_color_optical_frame \
  tracking_marker_frame:=optical_target

ros2 run tf2_ros static_transform_publisher \
  -0.062 -0.005 0  0.5 -0.5 -0.5 0.5  tcp camera_link



python3 /home/sungboo/ros2_ws/src/rb10_control/scripts/rb10_rosbag_to_hdf5.py \
  --folder /home/sungboo/ros2_ws/src/rb10_control/dataset/251017_kin_3 \
  --out /home/sungboo/ros2_ws/src/rb10_control/dataset/251017_kin_3.hdf5 \
  --no-normalize-actions --freedrive-only
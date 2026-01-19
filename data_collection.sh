# ============================================================
# 0. (Run once) eih calibration
ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false
ros2 launch realsense2_camera rs_launch.py 
ros2 launch easy_handeye2 calibrate.launch.py \
  calibration_type:=eye_in_hand \
  name:=my_eih_calib \
  robot_base_frame:=link0 \
  robot_effector_frame:=tcp \
  tracking_base_frame:=camera_color_optical_frame \
  tracking_marker_frame:=optical_target #####CHANGE THIS

# ============================================================
# 1. Setup
# setup robot and camera
# ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false
ros2 launch realsense2_camera rs_launch.py 
ros2 run tf2_ros static_transform_publisher -0.0495 -0.005 0  -0.5 0.5 -0.5 0.5  tcp_rbpodo camera_link
# 62mm - 25/2mm = 49.5mm

# apriltag
ros2 run apriltag_ros apriltag_node --ros-args \
    -r image_rect:=/camera/camera/color/image_rect_raw \
    -r camera_info:=/camera/camera/color/camera_info \
    --params-file /home/sungboo/ros2_ws/src/apriltag_ros/cfg/tags_36h11.yaml

# ============================================================
# 2. Data Collection
# topic pusblisher
python3 /home/sungboo/rb10_control/scripts/demo_recorder_bridge.py --keyboard --freedrive-on-start

# record ros2 bag
mkdir -p /home/sungboo/rb10_control/dataset/raw
cd /home/sungboo/rb10_control/dataset/raw
ros2 bag record -o demo_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /rb/joint_states \
  /rb/ee_pose \
  /rb/ee_wrench \
  /rb/stroke_event \
  /camera/camera/color/image_rect_raw \
  /camera/camera/color/camera_info

# ============================================================
# 3. Results Processing
# for image sticthing
cd /home/sungboo/rb10_control/dataset/raw
ros2 bag record -o res_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /camera/camera/color/image_rect_raw \
  /camera/camera/depth/image_rect_raw \
  /camera/camera/color/camera_info \
  /camera/camera/depth/camera_info

ros2 run image_view extract_images_sync --ros-args -p inputs:='[/camera/camera/color/image_rect_raw, /camera/camera/depth/image_rect_raw]'
# 안되면 https://github.com/MapIV/ros2_bag_to_image/tree/master
stitch img_dir/IMG*.jpg

# ============================================================
# 4. Replay and Collect
python3 /home/sungboo/rb10_control/scripts/rosbag_replay.py --bag /path/to/bag_folder
python3 /home/sungboo/rb10_control/scripts/demo_recorder_bridge.py
cd /home/sungboo/rb10_control/dataset/raw
ros2 bag record -o demo_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /rb/joint_states \
  /rb/ee_pose \
  /rb/ee_wrench

# ============================================================
# 5. Data Post-processing
# rosbag to hdf5
# python3 /home/sungboo/rb10_control/scripts/read_dataset.py
python3 /home/sungboo/rb10_control/scripts/rosbag_to_hdf5.py --folder "" --out "" --no-rgb
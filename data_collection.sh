# ============================================================
# 1. Setup
# setup robot and camera
ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false
ros2 launch realsense2_camera rs_launch.py 
ros2 run tf2_ros static_transform_publisher -0.0495 -0.005 0  0.5 -0.5 -0.5 0.5  tcp camera_link
# 62mm - 25/2mm = 49.5mm

# ============================================================
# 2. Data Collection
# topic pusblisher
python3 /home/sungboo/rb10_control/scripts/demo_recorder_bridge.py --freedrive-on-start --keyboard

# record ros2 bag
mkdir -p /home/sungboo/rb10_control/dataset/raw
cd /home/sungboo/rb10_control/dataset/raw
ros2 bag record -o demo_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /rb/joint_states \
  /rb/tcp_pose \
  /rb/ee_wrench \
  /rb/stroke_event \
  /camera/camera/color/image_raw \
  /camera/camera/camera_info

# ============================================================
# 3. Results Processing
# for image sticthing
cd /home/sungboo/rb10_control/dataset/raw
ros2 bag record -o res_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /camera/camera/color/image_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/camera_info

ros2 run image_view extract_images_sync --ros-args -p inputs:='[/camera/camera/color/image_raw, /camera/camera/aligned_depth_to_color/image_raw]'
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
  /rb/tcp_pose \
  /rb/ee_wrench

# ============================================================
# 5. Data Post-processing
# rosbag to hdf5
# python3 /home/sungboo/rb10_control/scripts/read_dataset.py
python3 /home/sungboo/rb10_control/scripts/rosbag_to_hdf5.py --folder "" --out "" --no-rgb
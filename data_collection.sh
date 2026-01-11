# setup robot and camera
ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false
ros2 launch realsense2_camera rs_launch.py 
ros2 run tf2_ros static_transform_publisher \
  -0.062 -0.005 0  0.5 -0.5 -0.5 0.5  tcp camera_link

# topic pusblisher
python3 /home/sungboo/rb10_control/scripts/rb10_demo_recorder_bridge.py

# record ros2 bag
cd /home/sungboo/rb10_control/dataset
ros2 bag record -o demo_$(date +%Y%m%d_%H%M%S) \
  /tf \
  /rb/joint_states \
  /rb/tcp_pose \
  /rb/ee_wrench \
  /rb/stroke_event \
  /camera/camera/color/image_raw \
  /camera/camera/camera_info

# ============================================================
# for image sticthing
cd /home/sungboo/rb10_control/dataset
ros2 bag record -o res_$(date +%Y%m%d_%H%M%S)\
  /tf \
  /camera/camera/color/image_raw \
  /camera/camera/aligned_depth_to_color/image_raw \
  /camera/camera/camera_info

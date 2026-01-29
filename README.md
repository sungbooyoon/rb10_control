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

---
# 1. Setup
## setup robot and camera
<!-- ros2 launch rbpodo_moveit_config moveit.launch.py use_fake_hardware:=false -->
ros2 launch realsense2_camera rs_launch.py
ros2 run tf2_ros static_transform_publisher -0.20317 0.009 -0.00405  0 0 0 1  tcp_rbpodo camera_link

## apriltag
ros2 run apriltag_ros apriltag_node --ros-args \
    -r image_rect:=/camera/camera/color/image_rect_raw \
    -r camera_info:=/camera/camera/color/camera_info \
    --params-file /home/sungboo/ros2_ws/src/apriltag_ros/cfg/tags_36h11.yaml

## (Run once) Apriltag detection test
python3 /home/sungboo/rb10_control/scripts/demo_recorder_bridge.py
cd /home/sungboo/rb10_control/data/raw
ros2 bag record -o apriltag_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /tf_static \
  /rb/ee_pose

---
# 2. Data Collection
## topic pusblisher
python3 /home/sungboo/rb10_control/scripts/demo_recorder_bridge.py --keyboard

## record ros2 bag
<!-- mkdir -p /home/sungboo/rb10_control/data/raw -->
cd /home/sungboo/rb10_control/data/raw
ros2 bag record -o demo_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /tf_static \
  /rb/joint_states \
  /rb/ee_pose \
  /rb/ee_wrench \
  /rb/stroke_event \
  /camera/camera/color/image_rect_raw \
  /camera/camera/color/camera_info

---
# 3. Results Processing
## for image sticthing
cd /home/sungboo/rb10_control/data/raw
ros2 bag record -o res_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /tf_static \
  /camera/camera/color/image_rect_raw \
  /camera/camera/depth/image_rect_raw \
  /camera/camera/color/camera_info \
  /camera/camera/depth/camera_info

python3 /home/sungboo/rb10_control/scripts/export_rgbd_from_rosbag.py --bag "" --hz 10
stitch --no-crop img_dir/IMG*.jpg 

---
# 4. Replay and Collect
python3 /home/sungboo/rb10_control/scripts/rosbag_replay.py --bag /path/to/bag_folder
python3 /home/sungboo/rb10_control/scripts/demo_recorder_bridge.py
cd /home/sungboo/rb10_control/data/raw
ros2 bag record -o demo_$(date +%Y%m%d_%H%M%S) -s mcap \
  /tf \
  /tf_static \
  /rb/joint_states \
  /rb/ee_pose \
  /rb/ee_wrench

---
# 5. Data Post-processing
## rosbag to hdf5
<!-- python3 /home/sungboo/rb10_control/scripts/read_dataset.py -->
python3 /home/sungboo/rb10_control/scripts/rosbag_to_hdf5.py \
  --folder /home/sungboo/rb10_control/data/demo_20260122 \
  --out /home/sungboo/rb10_control/data/demo_20260122_224+224.hdf5 \
  --rgb --image-resize 224 224

python3 /home/sungboo/rb10_control/scripts/fix_demo_index.py \
  --hdf5 /home/sungboo/rb10_control/data/demo_20260122_224+224.hdf5 \
  --out /home/sungboo/rb10_control/data/demo_20260122_224+224_new.hdf5 \
  --remove 180

python3 /home/sungboo/rb10_control/scripts/add_actions_to_dataset.py \
  --hdf5 /home/sungboo/rb10_control/data/demo_20260122_224+224_new.hdf5 \
  --out /home/sungboo/rb10_control/data/demo_20260122_224+224.hdf5 \
  --overwrite

python3 /home/sungboo/rb10_control/scripts/read_dataset.py \
  --hdf5 /home/sungboo/rb10_control/data/demo_20260122.hdf5 \
  --out-dir /home/sungboo/rb10_control/images/demo_20260122/

python3 /home/sungboo/rb10_control/robomimic/robomimic/scripts/get_dataset_info.py \
  --dataset /home/sungboo/rb10_control/data/demo_20260122_224+224.hdf5 \
  --verbose

## (Run Once) Tag 위치 추출
python3 /home/sungboo/rb10_control/scripts/extract_tf_from_rosbag.py \
  --bag /home/sungboo/rb10_control/data/apriltag/apriltag_20260122_172601 \
  --parent link0 \
  --tag_prefix tag36h11: \
  --out /home/sungboo/rb10_control/data/apriltag/apriltag_20260122_172601/link0_to_tag.csv

python3 /home/sungboo/rb10_control/scripts/cluster_apriltag.py \
  --csv /home/sungboo/rb10_control/data/apriltag/apriltag_20260122_172601/link0_to_tag.csv \
  --eps 0.003 \
  --min_samples 5 \
  --out /home/sungboo/rb10_control/data/apriltag/apriltag_20260122_172601/link0_to_tag_avg.csv

- Tag 1: 0.7792766396940909,-0.3408561185649196,1.087119146987467,0.5040227668956027,-0.49545032399749467,-0.5025265274518764,0.49795292559521853
- Tag 2: 0.7795913130268544,0.46453726735503503,1.0852591067208173,0.5030583310773759,-0.5039807608747278,-0.49834431333499923,0.4945590496274876

- 수정 Tag 1: 0.779, -0.340, 1.086, 0.5, -0.5, -0.5, 0.5
- 수정 Tag 2: 0.779, 0.460, 1.086, 0.5, -0.5, -0.5, 0.5


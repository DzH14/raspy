kalo mau colcon pake command di bawah,

colcon build --base-paths src

kalo mau run gui,

sourcing, terus

ros2 run barcode_hmi hmi_node

terus si backend cameranya,

buka terminal baru, sourcing, terus

ros2 run barcode_hmi live_backend --ros-args -p camera1_device:=2 -p camera2_device:=4

**2 ama 4 sesuaiin ama alamat si webcam yoii


// Step buat yg keintegrasiin ama arduino

cd ~/scannerBarcode

source install/setup.bash

ros2 run barcode_hmi hmi_node

// buka lagi terminal baru

cd ~/scannerBarcode

source install/setup.bash

ros2 run barcode_hmi polearm_node --ros-args -p camera1_device:=2 -p camera2_device:=4 -p serial_port:='/dev/ttyACM0'

// yg camera samaa, cek aee di v4l2-ctl --list-device, sesuaiin si addressnya, terus yg serial port buat arduino cek aee 'ls /dev/ttyACM*', sesuaiin, defaultnya kaya di atas

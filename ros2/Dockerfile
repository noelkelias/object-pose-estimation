FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    ffmpeg \
    xvfb \
    libgl1 \
    mesa-utils \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    ros-humble-tf2-ros \
    ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    "numpy>=1.24,<2" \
    "opencv-python-headless>=4.8,<5" \
    "open3d>=0.17,<0.20"

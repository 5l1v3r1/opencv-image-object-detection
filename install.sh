#!/bin/bash
sudo apt install python3 python3-pip
pip3 install -U tensorflow keras opencv-python
pip3 install imageai --upgrade
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
exit 0



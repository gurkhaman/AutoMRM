#!/bin/bash

apt-get update
apt-get upgrade -y
apt-get install -y git

pip install -r requirements.txt
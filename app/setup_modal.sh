#!/bin/bash
set -e
modal shell --image ubuntu:20.04 --gpu L4 --add-python 3.10 --volume moondream-vol --cmd "
apt update
apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev git
unset PYTHONPATH
unset PYTHONHOME
apt remove -y python3.9* || true
rm -f /usr/bin/python3.9*
rm -f /usr/local/bin/python3.9*
rm -rf /usr/lib/python3.9*
rm -rf /usr/local/lib/python3.9*
rm -rf /install*
rm -rf /pkg*

rm -f /usr/local/bin/python3
rm -f /usr/local/bin/python

cd /tmp
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
tar -xf Python-3.10.12.tgz
cd Python-3.10.12
./configure --enable-shared --enable-optimizations --prefix=/usr/local
make -j\$(nproc)
make altinstall
echo '/usr/local/lib' > /etc/ld.so.conf.d/python310.conf
ldconfig
rm -f /usr/bin/python*
rm -f /usr/bin/pip*

ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3
ln -sf /usr/local/bin/python3.10 /usr/local/bin/python
ln -sf /usr/local/bin/python3.10 /usr/bin/python3.10
ln -sf /usr/local/bin/python3.10 /usr/bin/python3
ln -sf /usr/local/bin/python3.10 /usr/bin/python
export PATH=/usr/local/bin:\$PATH
echo 'export PATH=/usr/local/bin:\$PATH' >> ~/.bashrc
/usr/local/bin/python3.10 -m ensurepip
ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3
ln -sf /usr/local/bin/pip3.10 /usr/bin/pip
pip install pyinstaller distro certifi

clear 

cd /mnt/moondream-vol
rm -rf moondream-station
git clone https://github.com/EthanReid/moondream-station.git

cd /mnt/moondream-vol/moondream-station/
git checkout main
git pull
ls
echo 'cloned git'

cd /mnt/moondream-vol/moondream-station/app
bash build.sh dev ubuntu
cd /mnt/moondream-vol/moondream-station
echo 'build files'

cd /mnt/moondream-vol/moondream-station/output
cp hypervisor.tar.gz /mnt/moondream-vol/moondream-station/
cp inference_bootstrap.tar.gz /mnt/moondream-vol/moondream-station/
cp moondream_station_ubuntu.tar.gz /mnt/moondream-vol/moondream-station/
cp moondream-cli.tar.gz /mnt/moondream-vol/moondream-station/
echo 'copied tar files'

exit
exec bash
"
echo "trying to download tar"
modal volume get moondream-vol /moondream-station/hypervisor.tar.gz
modal volume get moondream-vol /moondream-station/inference_bootstrap.tar.gz
modal volume get moondream-vol /moondream-station/moondream_station_ubuntu.tar.gz
modal volume get moondream-vol /moondream-station/moondream-cli.tar.gz

echo "Setup complete. Moondream Station files exported."
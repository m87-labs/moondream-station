#!/bin/bash
set -e

docker run --rm --gpus all -it ubuntu:20.04 bash -c "
apt update
apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev git

cd /tmp
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
tar -xf Python-3.10.12.tgz
cd Python-3.10.12

./configure --enable-shared --enable-optimizations --prefix=/usr/local

make -j\$(nproc)
make altinstall

echo '/usr/local/lib' > /etc/ld.so.conf.d/python310.conf
ldconfig

ln -sf /usr/local/bin/python3.10 /usr/bin/python3.10
ln -sf /usr/local/bin/python3.10 /usr/bin/python3
rm -f /usr/bin/python
ln -sf /usr/local/bin/python3.10 /usr/bin/python

export PATH=/usr/local/bin:\$PATH
echo 'export PATH=/usr/local/bin:\$PATH' >> ~/.bashrc

/usr/local/bin/python3.10 -m ensurepip
ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3
ln -sf /usr/local/bin/pip3.10 /usr/bin/pip

pip install pyinstaller distro certifi

cd ~
git clone <moondream-station-repo-url>
cd moondream-station/app
git checkout <branch-name>
bash build.sh dev ubuntu --build-clean

python --version
pip --version

exec bash
"
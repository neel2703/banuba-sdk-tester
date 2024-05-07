# BanubaSDK for Python

Package contains **PyBanubaSDK** for **Python 3.9** and two examples for different use cases:

* `rendering_example.py` - demonstrates how to apply the **effect** on the image (require **Nvidia GPU** with [EGL driver](https://www.nvidia.com/en-us/drivers/unix/) installed)
* `recognition_example.py` - demonstrates how to extract features from image (CPU only)

## Installation

```bash
cd PyBanubaSDK.1.12.0.cpython-39-x86_64-linux-gnu
pip install .
```

## Run

* Setup **client token** in `rendering_example.py` or `recognition_example.py`
* Run the example

## Troubleshooting

ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.13' not found

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11
```
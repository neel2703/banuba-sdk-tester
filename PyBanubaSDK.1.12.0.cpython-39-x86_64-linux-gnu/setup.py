from setuptools import setup
from distutils.sysconfig import get_config_var
import subprocess
import zipfile
import sys

VERSION='1.12.0'

def get_package_name(version):
    if not version:
        raise NameError("Package version can't be empty")

    var = get_config_var('EXT_SUFFIX')
    plat_config = var[:var.rfind('.')]
    suffix = plat_config + '.zip'
    package_name = 'PyBanubaSDK.' + version + suffix
    return package_name

def get_package_files(package):
    files_of_package = []
    with zipfile.ZipFile(package, 'r') as archive:
        files_of_package = [name for name in archive.namelist() if not name.endswith('/')]
        archive.extractall('.')
    return files_of_package

package_files = get_package_files(get_package_name(VERSION))
data = package_files
pep440_version = VERSION.split('-')[0]

setup(
    name='PyBanubaSDK',
    version=pep440_version,
    license="MIT",
    description="Banuba SDK for Python",
    long_description="",
    long_description_content_type="text/markdown",
    author="Banuba Limited",
    author_email="info@banuba.com.",
    url="https://banuba.com/",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    packages = [''],

    package_dir={'': '.'},
    package_data={'': data},

    install_requires=[
       'glfw',
       'Pillow',
       'numpy',
    ]
)


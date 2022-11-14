from setuptools import setup , find_packages
import pathlib

setup(
    name='VLAD',
    version='0.1.2',
    package_dir={"":"scr"},
    packages=find_packages(where='scr')
)
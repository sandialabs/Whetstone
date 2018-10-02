from setuptools import setup
from setuptools import find_packages

setup(name='whetstone', version='0.9.2', description='Provides extensions to keras to allow training deep spiking neural networks',packages=find_packages(exclude=("Release",)))

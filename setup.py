from setuptools import setup

with open('requirements.txt') as f:
    packages = f.read().splitlines()

setup(
    name="hand_detection_schredder_machine",
    version="1.0",
    description="Hand detection to stop schredder machine when hand gets too close to the machine",
    author="Modojojo",
    install_requires=packages
)
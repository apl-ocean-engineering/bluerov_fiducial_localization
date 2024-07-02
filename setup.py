from setuptools import find_packages, setup
import os
from glob import glob

package_name = "bluerov_fiducial_localization"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "cfg"),
            glob(os.path.join("cfg", "*")),
        ),
    ],
    install_requires=["setuptools", "blue_localization", "apriltag_msgs"],
    zip_safe=True,
    maintainer="Aaron Marburg",
    maintainer_email="amarburg@uw.edu",
    description="TODO: Package description",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bluerov_fiducial_localization = bluerov_fiducial_localization.ros2_entrypoint:entrypoint"
        ],
    },
)

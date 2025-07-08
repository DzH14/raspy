from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'barcode_hmi'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('barcode_hmi/polman_logo.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='HMI for barcode detection system',
    license='Apache-2.0',
    tests_require=['pytest'],
    package_data={'': ['*.msg']},  # <--- Tambah ini
    entry_points={
        'console_scripts': [
            'hmi_node = barcode_hmi.hmi_ros_node:main',
            'dummy_backend = barcode_hmi.dummy_backend_node:main',
            'live_backend = barcode_hmi.live_backend_node:main',
            'node_hmi = barcode_hmi.hmi_node:main',
            'polearm_node = barcode_hmi.polearm_node:main',
        ],
    },
)

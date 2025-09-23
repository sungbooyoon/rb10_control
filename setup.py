from setuptools import find_packages, setup

package_name = 'rb10_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sungboo',
    maintainer_email='sungboo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
			# ex_servo_spacenav = rb10_control.ex_servo_spacenav:main',
        ],
    },
	# scripts=['scripts/test_pub.py', 'scripts/quick_check.py'],
)

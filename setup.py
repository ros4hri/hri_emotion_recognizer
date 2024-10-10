from setuptools import find_packages, setup
package_name = 'hri_emotion_recognizer'

setup(
    name=package_name,
    version='1.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/ament_index/resource_index/pal_system_module',
         ['module/' + package_name]),
        ('share/ament_index/resource_index/pal_configuration.' + package_name,
            ['config/' + package_name]),
        ('share/' + package_name + '/config', ['config/00-defaults.yml']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/module',
         ['module/hri_emotion_recognizer_module.yaml']),
        ('share/' + package_name + '/launch', [
            'launch/emotion_recognizer.launch.py']),
        # ('share/' + package_name + '/test/data',
        #    [str(path) for path in Path('test/data').glob('**/*') if path.is_file()]),

    ],
    install_requires=['setuptools', 'hri_face_detect'],
    zip_safe=True,
    maintainer='Sara Cooper',
    maintainer_email='sara.cooper@pal-robotics.com',
    description='HRI Emotion recognizer',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hri_emotion_recognizer = hri_emotion_recognizer.hri_emotion_recognizer:main',
        ],
    },
)

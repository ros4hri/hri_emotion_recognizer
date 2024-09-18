# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.executors import SingleThreadedExecutor, TimeoutException
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from hri_msgs.msg import Expression
from datetime import timedelta, datetime
from rclpy.time import Time
import unittest
from cv_bridge import CvBridge
import cv2
import time
import csv
from hri import HRIListener
from hri_emotion_recognizer.hri_emotion_recognizer import NodeEmotionRecognizer
from hri_face_detect.node_face_detect import NodeFaceDetect
from pathlib import Path


def spin_some(executor, timeout=timedelta(seconds=10.)):
    start = datetime.now()
    cb_iter = executor._wait_for_ready_callbacks(timeout_sec=0.)
    while True:
        try:
            handler, *_ = next(cb_iter)
            handler()
            if handler.exception() is not None:
                raise handler.exception()
        except TimeoutException:
            elapsed = datetime.now() - start
            if elapsed > timeout:
                raise TimeoutException(
                    f'Time elapsed spinning {elapsed} with timeout {timeout}')
        except StopIteration:
            break


class TestHRIEmotionRecognizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()
        cls.emotion_recognizer_node = NodeEmotionRecognizer()
        cls.face_detect_node = NodeFaceDetect()
        cls.emotion_executor = SingleThreadedExecutor()
        cls.emotion_executor.add_node(cls.emotion_recognizer_node)
        cls.emotion_executor.add_node(cls.face_detect_node)
        cls.face_detect_node.trigger_configure()
        cls.emotion_recognizer_node.trigger_configure()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        cls.emotion_recognizer_node.destroy_node()
        cls.face_detect_node.destroy_node()
        rclpy.shutdown()
        return super().tearDownClass()

    def setUp(self) -> None:
        self.tester_node = rclpy.create_node('tester_node')
        self.clock_pub = self.tester_node.create_publisher(Clock, '/clock', 1)
        self.bridge = CvBridge()
        self.image_publisher = self.tester_node.create_publisher(
            Image, '/image', 10)
        self.subscriptions = {}
        self.expression_received = {}
        self.tester_executor = SingleThreadedExecutor()
        self.tester_executor.add_node(self.tester_node)
        self.hri_listener = HRIListener('tester_emotion_listener', False)
        self.face_detect_node.trigger_activate()
        self.emotion_recognizer_node.trigger_activate()
        return super().setUp()

    def tearDown(self) -> None:
        self.face_detect_node.trigger_deactivate()
        self.emotion_recognizer_node.trigger_deactivate()
        del self.hri_listener
        self.tester_node.destroy_node()
        return super().tearDown()

    def load_expected_emotions(self, csv_file_path):
        emotions = {}
        with open(csv_file_path / 'image_labels.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Assuming the CSV columns are 'image_filename' and 'emotion'
                image_filename = row['image_filename']
                emotion = row['emotion']
                emotions[image_filename] = emotion
        return emotions

    def publish_image_callback(self):
        if hasattr(self, 'current_image'):
            img = cv2.imread(str(self.current_image))
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            self.image_publisher.publish(img_msg)

    def spin(self, time_ns=None):
        if time_ns is not None:
            self.clock_pub.publish(
                Clock(clock=Time(nanoseconds=time_ns).to_msg()))
        spin_some(self.emotion_executor)
        self.hri_listener.spin_some(timedelta(seconds=5.))
        spin_some(self.tester_executor)

    def expression_callback(self, face_id):
        def callback(msg):
            self.expression_received[face_id] = msg
        return callback

    def test_emotion_recognition(self):
        # If running as python3 test_emotion_recog.py
        # csv_file_path = Path().cwd() / 'data'
        # If running as colcon test-result --verbose
        csv_file_path = Path().cwd() / 'test' / 'data'
        expected_expression_mapping = {
            'ANGRY': Expression.ANGRY,
            'HAPPY': Expression.HAPPY,
            'SURPRISED': Expression.SURPRISED,
            'SAD': Expression.SAD,
            'DISGUSTED': Expression.DISGUSTED,
            'NEUTRAL': Expression.NEUTRAL,
            'SCARED': Expression.SCARED,
        }
        self.expected_emotions = self.load_expected_emotions(csv_file_path)

        for img_path, expected_emotion in self.expected_emotions.items():
            found = False
            self.current_image = csv_file_path / img_path
            expected_emotion_enum = expected_expression_mapping[expected_emotion]

            # Start publishing the image
            self.image_timer = self.tester_node.create_timer(
                0.1, self.publish_image_callback)

            # Allow some time for the emotion recognizer to process
            timeout = 10  # seconds
            start_time = time.time()
            found = False

            while time.time() - start_time < timeout:
                self.spin()
                # Dynamically create subscriptions for detected face IDs
                for face_id in list(self.hri_listener.faces.keys()):
                    if face_id not in self.subscriptions:
                        self.subscriptions[face_id] = self.tester_node.create_subscription(
                            Expression,
                            f'/humans/faces/{face_id}/expression',
                            self.expression_callback(face_id),
                            10
                        )

                # Check if the expected emotion has been received
                for face_id, expression_msg in self.expression_received.items():
                    print(self.current_image)
                    print(expression_msg)
                    print(expected_emotion_enum)
                    if expression_msg.expression == expected_emotion_enum:
                        found = True
                        break
                if found:
                    break

            self.assertTrue(
                found, f"Expected emotion '{expected_emotion}' not found for image '{img_path}'")


if __name__ == '__main__':
    unittest.main()

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
from datetime import timedelta, datetime
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.parameter import Parameter
import unittest
from cv_bridge import CvBridge
import cv2
import csv
from hri import HRIListener
from hri_emotion_recognizer.hri_emotion_recognizer import NodeEmotionRecognizer
from hri_face_detect.node_face_detect import NodeFaceDetect
from pathlib import Path

NODE_RATE = 10.


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

        cls.face_detect_node = NodeFaceDetect()
        cls.face_detect_node.set_parameters([
            Parameter('deterministic_ids', Parameter.Type.BOOL, True),
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        cls.emotion_recognizer_node = NodeEmotionRecognizer()
        cls.emotion_recognizer_node.set_parameters([
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])

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
        cls.emotion_executor.shutdown()
        rclpy.shutdown()
        return super().tearDownClass()

    def setUp(self) -> None:
        self.tester_node = rclpy.create_node('tester_node')

        self.time = Time()

        self.clock_pub = self.tester_node.create_publisher(Clock, '/clock', 1)
        self.bridge = CvBridge()
        self.image_publisher = self.tester_node.create_publisher(
            Image, '/image', 10)

        self.subscriptions = {}

        self.expression_received = {}

        self.tester_executor = SingleThreadedExecutor()
        self.tester_executor.add_node(self.tester_node)

        self.hri_listener = HRIListener(
            'tester_emotion_listener', False, use_sim_time=True)

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

    def fastforward_time(self):
        self.time = self.time + Duration(seconds=(1/NODE_RATE))
        self.clock_pub.publish(Clock(clock=self.time.to_msg()))

    def spin(self):

        # publish the image
        spin_some(self.tester_executor)

        self.fastforward_time()

        # hri_face_detect detects the face
        spin_some(self.emotion_executor)
        spin_some(self.emotion_executor)

        # libhri publish the detected faces
        self.hri_listener.spin_some(timedelta(seconds=5.))

        self.fastforward_time()

        # hri_emotion_recognizer recognizes the emotion
        spin_some(self.emotion_executor)
        # libhri publish the recognized emotions
        self.hri_listener.spin_some(timedelta(seconds=5.))

    def test_emotion_recognition(self):
        # If running as python3 test_emotion_recog.py
        # csv_file_path = Path().cwd() / 'data'
        # If running as colcon test-result --verbose
        csv_file_path = Path().cwd() / 'test' / 'data'
        self.expected_emotions = self.load_expected_emotions(csv_file_path)

        self.spin()

        for img_path, expected_emotion in self.expected_emotions.items():
            self.current_image = csv_file_path / img_path

            print(
                f"Testing image {self.current_image} -> expected emotion {expected_emotion}")

            img = cv2.imread(str(self.current_image))
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')

            # need to publish the image multiple times to get the face detected
            # due to hri_face_detect tracking algorithm + the generation of the
            # cropped image. Then, need to spin several times to get all the messages
            # to the emotion recognizer amd back to libhri
            self.image_publisher.publish(img_msg)
            self.spin()
            self.image_publisher.publish(img_msg)
            self.spin()
            self.image_publisher.publish(img_msg)
            self.spin()
            self.image_publisher.publish(img_msg)
            self.spin()
            self.image_publisher.publish(img_msg)
            self.spin()
            self.spin()
            self.spin()
            self.spin()

            faces = self.hri_listener.faces

            self.assertTrue(
                len(faces) != 0, "No faces detected!")

            self.assertTrue(
                len(faces) == 1, "Only one face should be detected")

            self.assertIsNotNone(
                faces[list(faces.keys())[0]].expression, "No expression detected!")

            self.assertEqual(
                faces[list(faces.keys())[0]].expression.name, expected_emotion,
                f"Wrong expression detected for {self.current_image}!")


if __name__ == '__main__':
    unittest.main()

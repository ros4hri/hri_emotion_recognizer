#! /usr/bin/python3
# -*- coding: utf-8 -*-

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
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from hri_msgs.msg import Expression
from hri import HRIListener
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.lifecycle import Node, LifecycleState, TransitionCallbackReturn
from lifecycle_msgs.msg import State
import ament_index_python as aip
from pathlib import Path

# with opencv 4.5.4, what tiago uses, we need to use emotion-ferplus-8 model!

EMOTION_DICT = {
    0: Expression.NEUTRAL,
    1: Expression.HAPPY,
    2: Expression.SURPRISED,
    3: Expression.SAD,
    4: Expression.ANGRY,
    5: Expression.DISGUSTED,
    6: Expression.SCARED
}

AMENT_RESOURCE_TYPE = 'dnn_models.emotion_recognition'


class NodeEmotionRecognizer(Node):
    def __init__(self):
        super().__init__('hri_emotion_recognizer')
        self.declare_parameter(
            'emotion_model',
            'emotion-ferplus-8.onnx',
            ParameterDescriptor(description='ONNX model type')
        )

        self.last_recognised_emotions = {}

        # Start publishing diagnostics
        self.diag_pub = self.create_publisher(
            DiagnosticArray, '/diagnostics', 1)
        self.diag_timer = self.create_timer(1., self.publish_diagnostics)

        self.current_diagnostics_status = DiagnosticStatus.WARN
        self.current_diagnostics_message = 'Emotion recognizer is unconfigured'
        self.get_logger().info('State: Unconfigured.')

    def __del__(self):
        state = self._state_machine.current_state
        self.on_shutdown(LifecycleState(state_id=state[0], label=state[1]))

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.internal_cleanup()

        self.get_logger().info('State: Unconfigured.')
        return super().on_cleanup(state)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.model = "models/" + self.get_parameter('emotion_model').value

        self.current_diagnostics_status = DiagnosticStatus.WARN
        self.current_diagnostics_message = 'Emotion recognizer is configuring...'

        try:
            # Get all packages that provide the emotion recognition models
            available_resources = aip.get_resources(AMENT_RESOURCE_TYPE)
            if not available_resources:
                self.get_logger().error(
                    f"No resources found for ament_index type {AMENT_RESOURCE_TYPE}."
                    " Have you installed the emotion recognition models?")
                return TransitionCallbackReturn.FAILURE

            # Check each package for available models
            for package_name in available_resources.keys():
                resource_name, model_path = aip.get_resource(
                    AMENT_RESOURCE_TYPE, package_name)
                if not model_path:
                    self.get_logger().error(
                        f"No resource found in {package_name}")
                    continue  # Move to the next package

                if resource_name.endswith(self.model):
                    pkg_share_path = aip.get_package_share_directory(
                        package_name)
                    self.onnx_model_path = Path(pkg_share_path) / resource_name
                    break

            else:
                # If no matching model is found after the loop
                self.get_logger().error(
                    f"No matching model found for {self.model}")
                return TransitionCallbackReturn.FAILURE

            # Load the ONNX emotion model
            self.emotion_model = cv2.dnn.readNetFromONNX(
                str(self.onnx_model_path))
            self.get_logger().info(
                f"Loaded emotion model: {self.onnx_model_path}")

        except Exception as e:
            error_msg = f'Failed to load {self.onnx_model_path} model: {str(e)}'

            self.current_diagnostics_status = DiagnosticStatus.ERROR
            self.current_diagnostics_message = error_msg

            self.get_logger().error(error_msg)
            return TransitionCallbackReturn.FAILURE

        self.current_diagnostics_status = DiagnosticStatus.WARN
        self.current_diagnostics_message = (
            'Emotion recognizer is configured, yet inactive '
            '(lifecycle: inactive).'
        )
        self.get_logger().info('State: Inactive.')

        return super().on_configure(state)

    def internal_cleanup(self):

        self.destroy_timer(self.diag_timer)
        self.destroy_publisher(self.diag_pub)

    def internal_deactivate(self):

        self.destroy_timer(self.timer)
        for face_id in self.face_publishers:
            self.destroy_publisher(self.face_publishers[face_id])

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.internal_deactivate()

        self.current_diagnostics_status = DiagnosticStatus.WARN
        self.current_diagnostics_message = \
            'Emotion recognizer is configured, yet inactive (lifecycle: inactive).'

        self.get_logger().info('State: Inactive.')
        return super().on_deactivate(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.hri_listener = HRIListener('emotion_hri_listener')
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.face_publishers = {}
        self.current_diagnostics_status = DiagnosticStatus.OK
        self.current_diagnostics_message = \
            'Emotion Recognizer is active'
        self.get_logger().info('State: Active.')
        return super().on_activate(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        if state.state_id == State.PRIMARY_STATE_ACTIVE:
            self.internal_deactivate()
        if state.state_id in [State.PRIMARY_STATE_ACTIVE, State.PRIMARY_STATE_INACTIVE]:
            self.internal_cleanup()
        self.get_logger().info('State: Finalized.')
        return super().on_shutdown(state)

    def softmax(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()

    def postprocess(self, scores):

        prob = self.softmax(scores)
        prob = np.squeeze(prob)
        classes = np.argsort(prob)[::-1]
        return EMOTION_DICT[classes[0]], prob[classes[0]]

    def timer_callback(self):

        self.last_recognised_emotions = {}

        for face_id, face in self.hri_listener.faces.items():
            if face.cropped is not None:
                try:
                    img_face = cv2.resize(face.cropped, (64, 64))
                    img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
                    img_face = img_face.reshape(1, 1, 64, 64)

                    # Emotion detection
                    self.emotion_model.setInput(img_face)
                    output = self.emotion_model.forward()
                    emotion, confidence = self.postprocess(output[0])
                    # Prepare emotion result
                    emotion_result = {
                        'dominant_emotion': emotion,
                        # Assume the maximum score is the confidence
                        'confidence': confidence
                    }
                    # Only publish the emotion if the face_id is tracked
                    if face_id in self.hri_listener.faces:
                        self.last_recognised_emotions[face_id] = emotion_result.get(
                            "dominant_emotion")
                        self.publish_emotion(face_id, emotion_result)

                except Exception as e:
                    self.get_logger().error(
                        f"Failed to analyze emotion for face_id {face_id}: {e}")

    def publish_emotion(self, face_id, emotion_result):

        try:
            if face_id not in self.face_publishers:
                self.face_publishers[face_id] = self.create_publisher(
                    Expression, f'/humans/faces/{face_id}/expression', 10)

            msg = Expression()
            msg.expression = str(emotion_result.get(
                'dominant_emotion'))

            msg.confidence = float(str(emotion_result.get(
                'confidence')))
            self.face_publishers[face_id].publish(msg)

        except Exception as e:
            # Log the error and handle it gracefully
            self.get_logger().error(
                f"Failed to publish emotion for face_id {face_id}: {e}")

    def publish_diagnostics(self):
        arr = DiagnosticArray()
        msg = DiagnosticStatus(
            level=self.current_diagnostics_status,
            name='/social_perception/faces/hri_emotion_recognizer',
            message=self.current_diagnostics_message,
            values=[
                KeyValue(key="Module name", value="hri_emotion_recognizer"),
                KeyValue(key="Current lifecycle state",
                         value=self._state_machine.current_state[1]),
                KeyValue(key="Last recognised emotions",
                         value="; ".join(
                             f"face <{face_id}>: {emotion}"
                             for face_id, emotion
                             in self.last_recognised_emotions.items())),
            ],
        )

        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [msg]
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = NodeEmotionRecognizer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

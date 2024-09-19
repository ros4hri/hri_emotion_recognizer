^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_emotion_recognizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* rewrote the unit-test to use simpler logic and faster sim time
* fix ament_index model path computation bc hri_emotion_models are not ROS package anymore
* show last recognised emotion in diagnostics
* Impl emotion recognizer using ONNX models
  The models themselves are provided by external packages
  like hri_emotion_models
* Contributors: Sara Cooper, Séverin Lemaignan

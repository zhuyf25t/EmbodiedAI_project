# SO-101 模型来源

`so101.xml` 与 `assets/` 中的 STL 来自 MuJoCo Menagerie 的 `robotstudio_so101` 模型，提交为 `370c43965323c0944b433b202aac5058b5f907a9`。该模型基于 The Robot Studio 的 SO-101，随模型附上原始 Apache 2.0 许可。本项目按课程参考 URDF 的相机连杆变换安装 `wrist_cam`，使用 50° 垂直视场。[固定版本的模型与作者说明](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101)

`lift.xml`、`stack.xml`、`sort.xml` 为课程项目编写，各自定义任务方块，通过 `table.xml` 共享桌面、双臂安装、固定相机、灯光及仿真步长。

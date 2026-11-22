# SO-101 模型来源

`so101.xml` 与 `assets/` 中的 STL 来自 MuJoCo Menagerie 的 `robotstudio_so101` 模型，提交为 `370c43965323c0944b433b202aac5058b5f907a9`。该模型基于 The Robot Studio 的 SO-101，随模型附上原始 Apache 2.0 许可。本项目在同一腕部镜头位置增设了用于 Stack、Sort 的广角虚拟相机 `wrist_wide`。Lift 使用原模型的 `wrist_cam`。[固定版本的模型与作者说明](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/robotstudio_so101)

`lift.xml`、`stack.xml`、`sort.xml` 为课程项目编写，包含桌面、方块、固定相机、灯光及仿真步长。Sort 在同一桌面上装配两只机械臂。

RL with IsaacLab
================

.. |huggingface| image:: /_static/svg/hf-logo.svg
   :width: 16px
   :height: 16px
   :class: inline-icon

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/IsaacLab.png
   :align: center
   :width: 90%

   IsaacLab (image: `IsaacLab <https://isaac-sim.github.io/IsaacLab/develop/source/overview/reinforcement-learning/rl_existing_scripts.html#rlinf>`__).

`IsaacLab <https://isaac-sim.github.io/IsaacLab/develop/source/overview/reinforcement-learning/rl_existing_scripts.html#rlinf>`__ is NVIDIA's GPU-accelerated robot
learning simulator. You'll use RLinf to PPO-fine-tune GR00T N1.5 or OpenPI π₀.₅ on a
custom Franka cube-stacking task.

Overview
--------

SFT then PPO-fine-tune a VLA on the IsaacLab Franka stack-cube task.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      GR00T N1.5 · π₀.₅

   .. grid-item-card:: Algorithms
      :text-align: center

      PPO

   .. grid-item-card:: Tasks
      :text-align: center

      Franka stack-cube

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install → download Isaac Sim + an SFT model → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · Isaac Sim · an SFT checkpoint (steps below).

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Task
     - Description
   * - ``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0``
     - Stack the red block on the blue block, then stack the green block on the red block.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - RGB from a third-person camera and a wrist camera (256×256 by default) plus robot proprioception.
   * - Action
     - 7-dim continuous action: 3D position (x, y, z) + 3D rotation (roll, pitch, yaw) + gripper.
   * - Reward
     - Sparse 0/1 success reward.
   * - Prompt
     - ``Stack the red block on the blue block, then stack the green block on the red block.``

Installation
------------

.. include:: _setup_common.rst

**Docker image**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-isaaclab

   # For mainland China users:
   # infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-isaaclab

Switch to the matching virtual environment inside the image:

.. code:: bash

   # GR00T N1.5
   source switch_env gr00t

   # OpenPI π₀.₅
   # source switch_env openpi

**Custom environment**

Install the environment for the model you want to run:

.. code:: bash

   # Mainland China users can add --use-mirror.

   # GR00T N1.5
   bash requirements/install.sh embodied --model gr00t --env isaaclab
   source .venv/bin/activate

   # OpenPI π₀.₅
   # bash requirements/install.sh embodied --model openpi --env isaaclab
   # source .venv/bin/activate

Download Isaac Sim
~~~~~~~~~~~~~~~~~~

Download Isaac Sim 5.1.0 and initialize its shell environment:

.. code-block:: bash

   mkdir -p isaac_sim
   cd isaac_sim
   wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
   unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip
   rm isaac-sim-standalone-5.1.0-linux-x86_64.zip
   source ./setup_conda_env.sh

.. warning::

   Run ``source ./setup_conda_env.sh`` in every new terminal before launching IsaacLab.

Download the Model
------------------

Download the checkpoint for the model you plan to fine-tune.

**GR00T N1.5**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/RLinf/RLinf-Gr00t-SFT-Stack-cube

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download RLinf/RLinf-Gr00t-SFT-Stack-cube --local-dir RLinf-Gr00t-SFT-Stack-cube

**OpenPI π₀.₅**

.. code-block:: bash

   cd /path/to/save/model

   git lfs install
   git clone https://huggingface.co/YifWRobotics/RLinf-pi05-SFT-Stack-cube

   # Or use huggingface-hub:
   # export HF_ENDPOINT=https://hf-mirror.com
   pip install huggingface-hub
   hf download YifWRobotics/RLinf-pi05-SFT-Stack-cube --local-dir RLinf-pi05-SFT-Stack-cube

.. include:: _model_path.rst

The SFT checkpoints come from human demonstrations collected on the IsaacLab stack-cube
task. The dataset is available on |huggingface|
`IsaacLab-Stack-Cube-Data <https://huggingface.co/datasets/RLinf/IsaacLab-Stack-Cube-Data>`__.

Run It
------

Pick one config and launch training:

.. list-table::
   :header-rows: 1
   :widths: 26 46 28

   * - Model
     - Config
     - Command suffix
   * - GR00T N1.5
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml``
     - ``isaaclab_franka_stack_cube_ppo_gr00t``
   * - OpenPI π₀.₅
     - ``examples/embodiment/config/isaaclab_franka_stack_cube_ppo_openpi_pi05.yaml``
     - ``isaaclab_franka_stack_cube_ppo_openpi_pi05``

.. code:: bash

   # GR00T N1.5
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_gr00t

   # OpenPI π₀.₅
   bash examples/embodiment/run_embodiment.sh isaaclab_franka_stack_cube_ppo_openpi_pi05

What this does:

1. Starts the embodied training entrypoint with the selected Hydra config.
2. Creates Ray workers for the actor, rollout, and IsaacLab env components.
3. Runs PPO rollouts, computes sparse task rewards, and updates the VLA policy.

For standalone evaluation, use the unified :doc:`Evaluation CLI
<../../evaluations/reference/cli>` with config fallback and the same suffixes:
``isaaclab_franka_stack_cube_ppo_gr00t`` and
``isaaclab_franka_stack_cube_ppo_openpi_pi05``.

.. note::

   For GR00T, the default config separates env, rollout, and actor placement. For OpenPI,
   the default config collocates ``actor,env,rollout: all``. Tune
   ``cluster.component_placement``, ``rollout.pipeline_stage_num``, and
   ``actor.enable_offload`` for your GPU memory budget.

.. note::

   To add a custom IsaacLab task, implement it under
   ``rlinf/envs/sim/isaaclab/tasks/``, register it in ``rlinf/envs/sim/isaaclab/__init__.py``,
   then point ``init_params.id`` in an env config such as
   ``examples/embodiment/config/env/isaaclab_stack_cube.yaml`` at the new task id.

Load an N1.7 SFT Bundle
----------------------

Use your N1.7 ``LIBERO_PANDA`` SFT bundle to run a short stack-cube evaluation.
Set the source paths to your pinned checkouts before installing the model and
simulator combination:

.. code:: bash

   export GR00T_PATH=/path/to/Isaac-GR00T
   export ISAAC_LAB_PATH=/path/to/IsaacLab
   bash requirements/install.sh --torch 2.8.0 --python 3.11.14 embodied --model gr00t_n1d7 --env isaaclab

The installer reuses these directories. The validated integration uses
Isaac-GR00T ``23ace64f`` and IsaacLab ``4246b6b4``, with Python 3.11.14,
Torch 2.8.0+cu128, Transformers 4.57.3, FlashAttention 2.8.3 and Isaac Sim 5.1.0.
The project runtime lock and resource measurements are in
``embodied-template/docs/STACK_CUBE_N1_7_INTEGRATION.md``.

Select the SFT checkpoint and its local Cosmos snapshot:

.. code:: bash

   export REPO_PATH=$PWD
   export STACK_CUBE_SFT_BUNDLE=/path/to/stack-cube-n1.7-sft
   export COSMOS_BACKBONE_PATH=/path/to/models/nvidia/Cosmos-Reason2-2B
   export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
   python evaluations/eval_embodied_agent.py \
     --config-path ../examples/embodiment/config \
     --config-name isaaclab_franka_stack_cube_ppo_gr00t_n1d7

This configuration resets one environment and executes two 16-step chunks,
collects rewards and records MP4. The snapshot path must contain
``nvidia/Cosmos-Reason2`` because the pinned GR00T version selects its backbone
class using that string. IsaacLab scene assets may still use NVIDIA's asset service;
verify model loading in an isolated network namespace separately.

The converter selects principal axis-angle for the 8-dimensional state and
keeps both signed finger positions. Actions retain their relative IK units;
IsaacLab applies scale 0.5 to the six arm commands, and positive gripper opens.
The processor's valid horizon is 16 within the model's padded horizon of 40.
Use ``libero_sim`` with ``isaaclab_stack_cube`` and train-only min/max statistics.
The micro SFT checkpoint verifies the integration; evaluate task learning separately.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

The key signal is ``env/success_once``. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Enable video in the env config when you want rollout videos:

.. code:: yaml

   video_cfg:
     save_video: True
     info_on_video: True
     video_base_dir: ${runner.logger.log_path}/video/train

Enable W&B or SwanLab by adding logger backends:

.. code:: yaml

   runner:
     logger:
       logger_backends: ["tensorboard", "wandb"]  # or swanlab

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Model Stage
     - Success Rate
   * - GR00T N1.5 base model (no SFT)
     - 0.000
   * - GR00T N1.5 SFT model
     - 0.654
   * - GR00T N1.5 RL-tuned model (SFT + RL)
     - 0.897
   * - OpenPI π₀.₅ SFT model
     - 0.859
   * - OpenPI π₀.₅ RL-tuned model (SFT + RL)
     - 0.953

Acknowledgements
----------------

Credit to `Minghui Xu <https://github.com/smallcracker>`__ and
`Nan Yang <https://github.com/AquaSage18>`__ for the GR00T N1.5 example, and
`Yifan Wu <https://github.com/YifWRobotics>`__ for the OpenPI π₀.₅ example.

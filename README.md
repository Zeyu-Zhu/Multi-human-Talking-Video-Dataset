
<p align="center">
  <img src="assets/title_color.png" alt="Multi-human Interactive Talking Dataset" width="600">
</p>


<p align="center">
  <a href="">
    <img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages">
  </a>
  <a href="">
    <img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv">
  </a>
</p>



Official repository for *Muti-human Interactive Talking Dataset*



<p align="center"><img src="assets/motivation.png" width="800px"/><br> </p>

## 🔥 News
* We will release the code and dataset within 3 monthes.
* **[2025.5.11]** We initialize the Repo.

## 💾 MIT Dataset

<p align="center"><img src="assets/dataset.png" width="800px"/><br> </p>

We present a high-quality dataset for multi-human interactive talking video generation, comprising over \$12\$ hours of high-resolution conversational clips with diverse interaction patterns and approximately \$200\$ distinct identities from two talk shows. These curated videos form the core of our dataset, selected for their natural, engaging interactions, clear speaker dynamics, and avoidance of common visual phenomena in real-world videos such as camera motion, occlusions, and editing artifacts—ensuring clean yet diverse multi-speaker scenarios. This serves as an ideal starting resource for this challenging new task. Some samples are shown below:
<table>
  <tr>
    <td>
        <img src="assets/main_video/example_1.gif" width="250">
    </td>
    <td>
        <img src="assets/main_video/example_2.gif" width="250">
    </td>
    <td>
        <img src="assets/main_video/example_4.gif" width="250">
    </td>
     <td>
        <img src="assets/main_video/example_8.gif" width="250">
    </td>
    <td>
        <img src="assets/main_video/example_6.gif" width="250">
    </td>
    <td>
        <img src="assets/main_video/example_9.gif" width="250">
    </td>
  </tr>
</table>


To showcase the potential of our data collection pipeline and further increase dataset diversity, we expand our dataset with an additional 3 hours of video from the YouTube short film channel Omeleto, which features rich, natural interactions and diverse character dynamics.
While these additional videos do not contain shot transitions, they may include camera motion, occlusions, and other real-world artifacts. We believe this subset serves as a challenging and complementary test set that augments the cleaner, studio-style data in the main dataset.
Some samples are shown below:
<table>
  <tr>
    <td>
        <img src="assets/extended_video/vid_10_10.gif" width="250">
    </td>
    <td>
        <img src="assets/extended_video/vid_8_10.gif" width="250">
    </td>
    <td>
        <img src="assets/extended_video/vid_3_10.gif" width="250">
    </td>
  </tr>
</table>

## 🔧 Data Collection Pipeline
If you want to construct your own dataset. Please navigate to the following folder structure:
```
├── data_collection_pipeline/
│   ├── 4_spaiens/
│   │   ├── spaiens.txt
│   ├── whisperV/
│   │   ├── whisperV.txt
│   ├── 1_whisperV_inference.py
│   ├── 2_select_valid_clips.py
│   ├── 3_speaking_annotate.py
│   ├── requirement.txt
```
Follow the instructions to prepare the env:
```bash
cd data_collection_pipeline
conda create -n mit_data python=3.9
pip3 install -r requirement.txt
conda activate mit_data
```
Put your raw videos in a folder, e.g., `./videos_input`. Then run WhisperV inference (support multi-thread based on the number of your GPUs):
```bash
python 1_whisperV_inference.py --raw_data_path ./videos_input --save_root_path seg_output_path
```
Then select the valid clips and crop. You can specify the number of speakers in the video using the `num_people` in `2_select_valid_clips.py` configuration parameter.
```bash
python 2_select_valid_clips.py --seg_result_root_path seg_output_path --save_root_path select_video_clip_save_path
python 3_speaking_annotate.py --seg_result_root seg_output_path --vaild_video_root select_video_clip_save_path output_path --datasets_root your_dataset_save_root
```
Then please refer to [spaiens](https://github.com/facebookresearch/sapiens) for pose estimation.

Multi-human talking video generation is an exciting yet challenging task, we are looking forward to seeing the contribution of your data! Any request please email me at: zeyuzhu2077@outlook.com.

## 🏋️ Training code for CovOG

We also release the training code of our baseline model CovOG. Follow the instructions to prepare the env:
```bash
cd baseline_train
conda create -n mit_train python=3.10
pip3 install -r requirement.txt
conda activate mit_train
```
Download weights under the `./pretrained_weights` direcotry.
```bash
python tools/download_weights.py
```
First, construct the data config as follow according to your path under `./data_config`:
```
[
    {
        "image_dir": "",
        "pose_dir": "",
        "speak_info_path": "",
        "audio_embedding_path": ""
    },
]
```
Then run these lines to train the model for stage 1 and stage 2:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_stage_1.py --config configs/train/stage1_finetune.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train_stage_2_CovOG.py --config configs/train/stage2_finetune_audio.yaml
```
## 🙏 Acknowledgment

We refer to the following codebases when building our pipeline:

- [Sapiens](https://github.com/facebookresearch/sapiens): For body pose estimation annotation.
- [WhisperV](https://github.com/showlab/whisperV): For audio-visual speaker activity annotation.
- [Hallo2](https://github.com/fudan-generative-vision/hallo2): For auido control module.
- [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone): For baseline model training.

We sincerely thank the authors of these works for their valuable open-source contributions, which greatly facilitated our research and development.

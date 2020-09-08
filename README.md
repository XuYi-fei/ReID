# Smart-Traffic-Vehicle-ReID

Vehicle ReID pipeline in smart traffic project



# Installation

#### 1.Install the mmcv:

You must install from Tsinghua source:

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mmcv-full==1.0.5
```

---

### 2.Install the mmdetection and **AICITY2020_DMT_VehicleReID**:

#### Requirements:

- Linux or macOS (Windows is not currently officially supported)

- Python 3.6+
- PyTorch 1.3+ 
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv](https://github.com/open-mmlab/mmcv) (Already satisfield)

We use torch==1.4.0+cu100 and torchvision==0.5.0+cu100 here

a.  Clone the mmdetection repository.

```shell
git clone https://github.com/XuYi-fei/ReID.git
cd mmdetection
```

b. Install build requirements and then install mmdetection.
(We install our forked version of pycocotools via the github repo instead of pypi
for better compatibility with our repo.)

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

---



Install dependencies:

- [pytorch>=1.1.0](https://pytorch.org/)
- python>=3.5
- torchvision
- [yacs](https://github.com/rbgirshick/yacs)
- cv2
- python-Levenshetein
- tqdm

Also the weight file should be placed under **/mmdetection/weight/**, if you don't have this file, create one.

---

# Get Started

- Remember to create a file named **test_output** to store the results, it should be at the same position as **test_img** 

- ```
  python main.py 
  ```

  to start test



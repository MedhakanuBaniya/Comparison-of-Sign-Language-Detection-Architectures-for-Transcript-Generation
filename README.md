# Comparison-of-Sign-Language-Detection-Architectures-for-Transcript-Generation
Conducted a study on multimodal deep learning for sign language recognition and translation. Integrated YOLOv8 for gesture recognition with vision transformers for translating gestures into text. Achieved high accuracy in both detection and translation, enhancing communication tools for the hearing-impaired.
# NLP_Signlanguagedetection



### Link to Google Colab is:
https://colab.research.google.com/drive/1TMZlb_Go3yltlDF9LnrqdCk85UtRy7dB?usp=sharing#scrollTo=DJn80zjQ8Boz

## Link to Document:
[NLP Report](https://github.com/iambikash/NLP_Signlanguagedetection/blob/main/NLP%20REPORT%20ON%20SIGN%20LANGUAGE%20.pdf)

#**Importing the Required Libraries**

```python
import os

```


```python
import glob
```


```python
from IPython.display import Image, display
```

# **In the First Step, We need to check whether we have access to the GPU or not**


```python
!nvidia-smi
```

    Mon Apr 29 19:20:29 2024       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
    | N/A   56C    P8              10W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
                                                                                             
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+



```python
HOME = os.getcwd()
```


```python
print(HOME)
```

    /content


#**Installing Ultralytics using Pip Install**


```python
!pip install ultralytics -q
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m755.0/755.0 kB[0m [31m15.7 MB/s[0m eta [36m0:00:00[0m
    [?25h

##Checking whether YOLOv8 is Installed and its working Fine


```python
import ultralytics
```


```python
ultralytics.checks()
```

    Ultralytics YOLOv8.2.5 🚀 Python-3.10.12 torch-2.2.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
    Setup complete ✅ (8 CPUs, 51.0 GB RAM, 35.2/201.2 GB disk)


#**Importing the American Sign Language Alphabets Dataset from RoboFlow**

https://universe.roboflow.com/0113-jadeniphit-ratiphongphan-kwpn6/sign-language-ca8mv



```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')


```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
%cd /content/drive/MyDrive/
```

    /content/drive/MyDrive


#**Train the YOLOv8 Model on the Custom Dataset**


```python
%cd {HOME}
```

    /content



```python
# Define the dataset information
data = """
path: /content/drive/MyDrive/Sign language dataset/FSL Dataset / # dataset root dir
train: images/train  # train images (relative to 'path') 4 images
val: images/val  # val images (relative to 'path') 4 images
test:  # test images (optional)
nc:  48 # number of classes
#classes: labels/train/class.txt  # class names
names: ["A", "Afternoon", "Are", "B", "Boss", "C", "D", "E", "F", "Father", "G", "Give", "Good", "H", "Hello", "How", "I", "I love you", "J", "K", "L", "M", "Me", "Mine", "Morning", "Mother", "N", "Night", "O", "P", "Q", "Quiet", "R", "S", "See", "Serious", "T", "Think", "This", "U", "V", "W", "Wait", "Water", "X", "Y", "You", "Z"]

"""

# Write the dataset information to data.yaml file
with open('/content/data.yaml', 'w') as f:
    f.write(data)

```


```python
import yaml
with open("/content/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
```


```python
num_classes
```




    '48'




```python
from ultralytics import YOLO
```


```python


# Load a model
model = YOLO('yolov8l.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/content/data.yaml", epochs=5, imgsz=640)
```

    Downloading https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt to 'yolov8l.pt'...


    100%|██████████| 83.7M/83.7M [00:00<00:00, 243MB/s]


    Ultralytics YOLOv8.2.5 🚀 Python-3.10.12 torch-2.2.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
    [34m[1mengine/trainer: [0mtask=detect, mode=train, model=yolov8l.pt, data=/content/data.yaml, epochs=5, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=train5, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train5
    Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf'...


    100%|██████████| 755k/755k [00:00<00:00, 67.7MB/s]

    Overriding model.yaml nc=80 with nc=48
    
                       from  n    params  module                                       arguments                     
      0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
      1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
      2                  -1  3    279808  ultralytics.nn.modules.block.C2f             [128, 128, 3, True]           
      3                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
      4                  -1  6   2101248  ultralytics.nn.modules.block.C2f             [256, 256, 6, True]           
      5                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
      6                  -1  6   8396800  ultralytics.nn.modules.block.C2f             [512, 512, 6, True]           
      7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              


    


      8                  -1  3   4461568  ultralytics.nn.modules.block.C2f             [512, 512, 3, True]           
      9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  3   4723712  ultralytics.nn.modules.block.C2f             [1024, 512, 3]                
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  3   1247744  ultralytics.nn.modules.block.C2f             [768, 256, 3]                 
     16                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  3   4592640  ultralytics.nn.modules.block.C2f             [768, 512, 3]                 
     19                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  3   4723712  ultralytics.nn.modules.block.C2f             [1024, 512, 3]                
     22        [15, 18, 21]  1   5619808  ultralytics.nn.modules.head.Detect           [48, [256, 512, 512]]         
    Model summary: 365 layers, 43666848 parameters, 43666832 gradients, 165.6 GFLOPs
    
    Transferred 589/595 items from pretrained weights
    [34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/detect/train5', view at http://localhost:6006/
    Freezing layer 'model.22.dfl.conv.weight'
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks with YOLOv8n...
    [34m[1mAMP: [0mchecks passed ✅


    [34m[1mtrain: [0mScanning /content/drive/MyDrive/Sign language dataset/FSL Dataset /labels/train.cache... 4430 images, 34 backgrounds, 0 corrupt: 100%|██████████| 4430/4430 [00:00<?, ?it/s]


    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))


    /usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()
    [34m[1mval: [0mScanning /content/drive/MyDrive/Sign language dataset/FSL Dataset /labels/val.cache... 32 images, 0 backgrounds, 0 corrupt: 100%|██████████| 32/32 [00:00<?, ?it/s]


    Plotting labels to runs/detect/train5/labels.jpg... 
    [34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    [34m[1moptimizer:[0m AdamW(lr=0.000192, momentum=0.9) with parameter groups 97 weight(decay=0.0), 104 weight(decay=0.0005), 103 bias(decay=0.0)
    [34m[1mTensorBoard: [0mmodel graph visualization added ✅
    Image sizes 640 train, 640 val
    Using 8 dataloader workers
    Logging results to [1mruns/detect/train5[0m
    Starting training for 5 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size


            1/5      11.2G      1.125      2.275       1.46         20        640: 100%|██████████| 277/277 [05:20<00:00,  1.16s/it]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:01<00:00,  1.69s/it]

                       all         32         32      0.209      0.265      0.321      0.246


    


    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size


            2/5        11G      1.053     0.9452      1.393         24        640: 100%|██████████| 277/277 [03:43<00:00,  1.24it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  1.49it/s]

                       all         32         32      0.249      0.156      0.227      0.165


    


    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size


            3/5        11G      1.019     0.7634      1.365         26        640: 100%|██████████| 277/277 [03:42<00:00,  1.24it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  1.48it/s]

                       all         32         32      0.182      0.266      0.301      0.215


    


    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size


            4/5      10.9G     0.9745     0.6348      1.339         30        640: 100%|██████████| 277/277 [03:41<00:00,  1.25it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  1.48it/s]

                       all         32         32      0.192       0.25      0.269      0.188


    


    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size


            5/5      10.9G     0.9096     0.5459       1.29         22        640: 100%|██████████| 277/277 [03:41<00:00,  1.25it/s]
                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  1.46it/s]

                       all         32         32      0.176      0.259      0.272      0.222


    


    
    5 epochs completed in 0.342 hours.
    Optimizer stripped from runs/detect/train5/weights/last.pt, 87.7MB
    Optimizer stripped from runs/detect/train5/weights/best.pt, 87.7MB
    
    Validating runs/detect/train5/weights/best.pt...
    Ultralytics YOLOv8.2.5 🚀 Python-3.10.12 torch-2.2.1+cu121 CUDA:0 (Tesla T4, 15102MiB)
    Model summary (fused): 268 layers, 43643616 parameters, 0 gradients, 165.0 GFLOPs


                     Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  1.42it/s]


                       all         32         32       0.21      0.267      0.321      0.246
                         A         32          1          0          0          0          0
                         B         32          1          0          0          0          0
                      Boss         32          1       0.14          1      0.995      0.597
                         C         32          1          0          0          0          0
                    Father         32          1       0.28          1      0.995      0.995
                         G         32          1          0          0          0          0
                      Good         32          1          0          0          0          0
                         H         32          1          0          0          0          0
                         I         32          1          0          0          0          0
                I love you         32          1      0.177       0.53      0.332      0.298
                         J         32          1          0          0          0          0
                         K         32          1          0          0          0          0
                         L         32          1          0          0          0          0
                         M         32          1          0          0          0          0
                        Me         32          1      0.778          1      0.995      0.597
                      Mine         32          1      0.534          1      0.995      0.796
                    Mother         32          1          1          0      0.995      0.895
                         N         32          1          0          0          0          0
                         Q         32          1          0          0          0          0
                     Quiet         32          1          1          0          0          0
                         S         32          1          0          0          0          0
                   Serious         32          1      0.416          1      0.995      0.497
                         T         32          1          0          0          0          0
                     Think         32          1      0.526          1      0.995      0.895
                      This         32          1      0.538          1      0.995      0.895
                         V         32          1          0          0          0          0
                         W         32          1          0          0          0          0
                     Water         32          1          1          0      0.995      0.697
                         X         32          1          0          0          0          0
                         Y         32          1          0          0          0          0
                       You         32          1      0.323          1      0.995      0.697
                         Z         32          1          0          0          0          0
    Speed: 0.2ms preprocess, 17.4ms inference, 0.0ms loss, 0.9ms postprocess per image
    Results saved to [1mruns/detect/train5[0m



```python
%cd runs/detect/train5/weights
```

    /content/drive/MyDrive/runs/detect/train5/weights



```python
new_model = YOLO('/content/best (1).pt')
```


```python
results = new_model('/content/Test Folder')
```

    
    image 1/3 /content/Test Folder/(1) Hello_test.jpg: 640x640 1 Hello, 61.8ms
    image 2/3 /content/Test Folder/(2) Morning_test.jpg: 640x640 2 Mornings, 61.8ms
    image 3/3 /content/Test Folder/(3) Mother_test.jpg: 640x640 1 Mother, 61.7ms
    Speed: 1.8ms preprocess, 61.8ms inference, 1.8ms postprocess per image at shape (1, 3, 640, 640)


Let us test the same figure using vision

#Lets compare the result with vision transformer


```python
import matplotlib.pyplot as plt
import torch
import torchvision
import torchinfo
from torch import nn
from torchvision import transforms
#from helper_functions import set_seeds
```


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```




    'cuda'




```python
# 1. Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# 2. Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# 4. Change the classifier head
class_names = ['Afternoon','Are','Boss','Father','Give','Hello','How','I Love you','Me','Mine','Morning','Mother','Night','See','Serious','Think','This','Wait','Water','You']

#set_seeds()
pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
```


```python
from torchinfo import summary

# Print a summary using torchinfo (uncomment for actual output)
summary(model=pretrained_vit,
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)
```




    ============================================================================================================================================
    Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
    ============================================================================================================================================
    VisionTransformer (VisionTransformer)                        [32, 3, 224, 224]    [32, 20]             768                  Partial
    ├─Conv2d (conv_proj)                                         [32, 3, 224, 224]    [32, 768, 14, 14]    (590,592)            False
    ├─Encoder (encoder)                                          [32, 197, 768]       [32, 197, 768]       151,296              False
    │    └─Dropout (dropout)                                     [32, 197, 768]       [32, 197, 768]       --                   --
    │    └─Sequential (layers)                                   [32, 197, 768]       [32, 197, 768]       --                   False
    │    │    └─EncoderBlock (encoder_layer_0)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_1)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_2)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_3)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_4)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_5)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_6)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_7)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_8)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_9)                   [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_10)                  [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    │    └─EncoderBlock (encoder_layer_11)                  [32, 197, 768]       [32, 197, 768]       (7,087,872)          False
    │    └─LayerNorm (ln)                                        [32, 197, 768]       [32, 197, 768]       (1,536)              False
    ├─Linear (heads)                                             [32, 768]            [32, 20]             15,380               True
    ============================================================================================================================================
    Total params: 85,814,036
    Trainable params: 15,380
    Non-trainable params: 85,798,656
    Total mult-adds (G): 5.52
    ============================================================================================================================================
    Input size (MB): 19.27
    Forward/backward pass size (MB): 3330.74
    Params size (MB): 229.25
    Estimated Total Size (MB): 3579.26
    ============================================================================================================================================




```python
# Setup directory paths to train and test images
train_dir = '/content/drive/MyDrive/NLP_Dataset_Final1/NLP_Dataset_Final/train'
test_dir = '/content/drive/MyDrive/NLP_Dataset_Final1/NLP_Dataset_Final/test'
```


```python
# Get automatic transforms from pretrained ViT weights
pretrained_vit_transforms = pretrained_vit_weights.transforms()
print(pretrained_vit_transforms)
```

    ImageClassification(
        crop_size=[224]
        resize_size=[256]
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        interpolation=InterpolationMode.BILINEAR
    )



```python
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):

  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)


  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
```


```python
# Setup dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_dir=train_dir,
                                                                                                     test_dir=test_dir,

                                                                                                     transform=pretrained_vit_transforms,
                                                                                                     batch_size=32) # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)
```


```python
"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import os
import zipfile
from pathlib import Path
import requests
import os



# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".

    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()
    return class_names[target_image_pred_label]

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
```


```python
"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results
```


```python

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                             lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the classifier head of the pretrained ViT feature extractor model
# set_seeds()
pretrained_vit_results = train(model=pretrained_vit,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=5,
                                      device=device)
```


      0%|          | 0/5 [00:00<?, ?it/s]


    /usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
      self.pid = os.fork()


    Epoch: 1 | train_loss: 0.8389 | train_acc: 0.8545 | test_loss: 0.2084 | test_acc: 0.9922
    Epoch: 2 | train_loss: 0.1309 | train_acc: 0.9952 | test_loss: 0.0877 | test_acc: 1.0000
    Epoch: 3 | train_loss: 0.0674 | train_acc: 0.9980 | test_loss: 0.0519 | test_acc: 0.9989
    Epoch: 4 | train_loss: 0.0448 | train_acc: 0.9983 | test_loss: 0.0376 | test_acc: 1.0000
    Epoch: 5 | train_loss: 0.0333 | train_acc: 0.9986 | test_loss: 0.0287 | test_acc: 1.0000



```python
import os
from PIL import Image
class_information = []
folder_path = '/content/Test Folder/'  # replace with your folder path
output_folder = '/content/Output Folder/'  # replace with your output folder path

# List all files in the directory
files = os.listdir(folder_path)

# Iterate over all files
for filename in files:
    # Check if the file is an image (you may need to adjust this condition based on your files)
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct full file path
        file_path = os.path.join(folder_path, filename)

        # Open the image file
        img = Image.open(file_path)

        # Set the new width and height
        new_width = 224
        new_height = 224

        # Resize the image
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # Construct output file path
        output_file_path = os.path.join(output_folder, 'output_' + filename)

        # Save the resized image
        img.save(output_file_path)

        # Predict on custom image
        class_info = pred_and_plot_image(model=pretrained_vit,
                                         image_path=output_file_path,
                                         class_names=class_names)
        class_information.append(class_info)
```

    <ipython-input-152-da4aaa4a73c2>:25: DeprecationWarning: ANTIALIAS is deprecated and will be removed in Pillow 10 (2023-07-01). Use LANCZOS or Resampling.LANCZOS instead.
      img = img.resize((new_width, new_height), Image.ANTIALIAS)



    
![png](https://github.com/iambikash/NLP_Signlanguagedetection/blob/main/NLP_Sign_Language_Detection_using_YoloV8_%26_Vision_Transformer_and_translating_to_Nepali_39_2.png)
    



    
![png](https://github.com/iambikash/NLP_Signlanguagedetection/blob/main/NLP_Sign_Language_Detection_using_YoloV8_%26_Vision_Transformer_and_translating_to_Nepali_39_3.png)
    



```python
class_information
```




    ['Hello', 'Mother', 'Morning']



#Let's translate the generated classes to a Nepali Language


```python
!huggingface-cli login
```

    
        _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
        _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
        _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
        _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
        _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
    
        A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
        Setting a new token will erase the existing one.
        To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
    Token: 
    Add token as git credential? (Y/n) Y
    Token is valid (permission: read).
    Your token has been saved in your configured git credential helpers (store).
    Your token has been saved to /root/.cache/huggingface/token
    Login successful



```python
!pip install git+https://github.com/huggingface/transformers.git -q
```

      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
      Preparing metadata (pyproject.toml) ... [?25l[?25hdone



```python
!pip install --upgrade transformers -q
```


```python
!pip uninstall transformers -q
!pip install transformers -q
```

    Proceed (Y/n)? Y
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.0/9.0 MB[0m [31m71.8 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
from transformers import pipeline
```


```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

# available models: 'facebook/nllb-200-distilled-600M', 'facebook/nllb-200-1.3B', 'facebook/nllb-200-distilled-1.3B', 'facebook/nllb-200-3.3B'
model_name = 'facebook/nllb-200-distilled-600M'

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

source = 'eng_Latn' # English
target = 'npi_Deva' # Nepali

translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=source, tgt_lang=target)
```


```python
pip install -U tensorflow==2.10 -q
```


```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# available models: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2', 'distilgpt2-medium', 'distilgpt2-large', 'distilgpt2-
```


```python

# Load the pre-trained model and tokenizer
model_name_LM = "gpt2-xl"
tokenizer_LM = AutoTokenizer.from_pretrained(model_name_LM)
model_LM = AutoModelForCausalLM.from_pretrained(model_name_LM)
```


```python
print(class_information)
prompt = ' '.join(class_information)
```

    ['Morning', 'Hello', 'Mother']



```python
output = translator(prompt, max_length=400)

translated_text = output[0]['translation_text']
print(translated_text)
```

    नमस्कार आमा



```python
input_ids = tokenizer_LM.encode(prompt, return_tensors="pt")

# Generate a continuation of the prompt
output_length = 100
generated_sequence = model_LM.generate(
    input_ids,
    max_length=input_ids.shape[-1] + output_length,
    pad_token_id=tokenizer_LM.pad_token_id,
    return_dict_in_generate=True,
    output_scores=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
    do_sample=True,
)

# Decode the generated sequence
generated_text = tokenizer_LM.decode(generated_sequence.sequences[0][:200])
generated_text=generated_text[0:generated_text.find('.')+1]
output = translator(generated_text, max_length=400)

translated_text = output[0]['translation_text']

```

    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.



```python
print('Input Sequence: ')
print(prompt)
print('*****'*5)
print('Generated Text: ')
print(generated_text)
print('*****'*5)
print('Translated Text in Nepali: ')
print(translated_text)
```

    Input Sequence: 
    Morning Hello Mother
    *************************
    Generated Text: 
    Morning Hello Mother, You have no idea how hard it was to get ready for this meeting.
    *************************
    Translated Text in Nepali: 
    नमस्कार आमा, तपाईलाई थाहा छैन यो बैठकको लागि तयारी गर्न कति गाह्रो थियो।


#                                               Thank you

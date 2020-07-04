# Age-Gender-Smile-Multitask-Learning
---
In this paper, I propose an architecture of Convolutional Neural Network (CNN) which can jointly learn representations for three tasks: smile detection,  gender and age classification. 

My model is based on the BKNet architecture. The method used hard parameter sharing, which to reduce the overfit of training separate task.
The proposed network takes input from multiple data sources, data were flow through CNN Shared Block which learns joint representations for
all tasks from all the sources of data. After the shared block, we separate network into three difference tasks. Each branch then learns
task-specific features and has its own loss calculation method.

<p align="center">
  <img width="400" height="500" src="https://user-images.githubusercontent.com/48142689/76413110-69a0a980-63c7-11ea-8ff1-2722fef44186.jpg">
</p>

## Quick start
```
python demo.py -i image_path      # estimate image
python demo.py -v video_path      # estimate video
python demo.py                    # video stream
```

## Dependencies
* python 3.7+
* tensorflow
* numpy
* opencv3.x
* **MTCNN** for face detection

## Datasets
Firstly, I prepare the training data by merge three datasets. I try to keep the number of training data for each task equally to help have the same impact of each dataset on the model
* [IMDB-WIKI Gender Datasets](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) 
* [IMDB-WIKI Age Datasets](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
* [GENKI-4K Smile Datasets](https://inc.ucsd.edu/mplab/wordpress/index.html%3Fp=398.html)
## Trainning
Run ```trainning.ipynb```. Change your datasets folder links, training parameters in ```config.py```.

## Evaluation
Run ```testing.ipynb``` to see result on the test datasets

## Result
|  Branch  | Train | Test |
|-----|-------|-------|
| `AGE`    | 66.06%  |  61.36% |
| `GENDER` | 97.01%  |  93.58% |
| `SMILE`  | 99.53%  |  92.80% |

<p align="center">
<img  width="1400" height="400" src="https://user-images.githubusercontent.com/48142689/76433597-1e968e80-63e7-11ea-8104-79194c32a411.jpg">
</p>

## Another version
I also build age-gender-estimation based on efficientnets. If you want, please see [age-gender-estimation](https://github.com/buiquangmanhhp1999/age_gender_estimation)

## Reference
* [Effective Deep Multi-source Multi-task Learning Frameworks for SmileDetection, Emotion Recognition and Gender Classiﬁcation](https://www.researchgate.net/publication/328586470_Effective_Deep_Multi-source_Multi-task_Learning_Frameworks_for_Smile_Detection_Emotion_Recognition_and_Gender_Classification?fbclid=IwAR0Mw11DfcFSOfpqFLp4rcHuVG06TC7KG6C9mrOHXktH_8slFvSCsBMtlMk)
* Dinh Viet Sang, Le Tran Bao Cuong, Pham Thai Ha, Multi-task learning for smile detection, emotion recognition and gender classification, December 2017

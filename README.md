# MATL
 
This is the basic implementation of our submission in ISSTA 2023: **Mapping APIs in Dynamic-typed Programs by Leveraging Transfer Learning**.
- [MATL](#matl)
  * [Description](#description)
  * [Project Structure](#project-structure)
  * [Datasets](#datasets)
  * [Reproducibility:](#reproducibility-)
    + [Environment:](#environment-)
    + [Preparation](#preparation)
  * [Contact](#contact)

## Description

`Matl` is a novel appraoch which leverages the transfer learning technique to learn the semantic embeddings of source code implementations from large-scale open-source repositories and then transfers the learned model to facilitate the mapping of APIs.
Firstly, we conduct an extensive study to explore their performance for mapping APIs in dynamic-typed languages. `MATL` is inspired by the insights of the study. In particular, the source code implementations of APIs can significantly improve the effectiveness of API mapping.
To evaluation study for the performance of Matl with state-of-the-art approaches demonstrate that Matl is indeed effective as it improves the state-of-the-art approach.

## Project Structure

```
├─approaches  # MATL main entrance.
|	├─dataset  # contains the training dataset.
|	├─vocab  # contains the extracted vocabulary.
|	├─models.py  # the MATL fine tune model structure.
|	├─main.py  # the MATL entrance.
|	├─config      # Configuration for MATL fine tuning
|	├─preTrain_model      # pretrained models
|	├─train.py      # the train file
|	├─eval.py      # the test file
|	├─utils.py      # the util
├─pretrain      # pretrain code
├─Dataset    # the dataset on DL Framework and Java2swift.
|	├─DL2DL      # Tensorflow, Torch, CNTK && MXNet
|		├─mappings.xlsx      # the mapping relationship from one Framework to Another
|		├─"frameworkname"_sig.txt      # the pre-processed signatures of framework APIs
|		├─"frameworkname"_desc.txt      # the pre-processed document of framework APIs
|		├─"frameworkname"_name.txt      # the pre-processed name of framework APIs
|	├─java2swift      # dataset of java2swift
|		├─mappings.xlsx      # the mapping relationship of java to swift
|		├─'*.txt'      # the pre-processed results of corresponding APIs
├─utils
├─logs        
├─datasets    
├─models      # Attention-based GRU and HDBSCAN Clustering.
├─module      # Anomaly detection modules, including classifier, Attention, etc.
├─outputs           
├─parsers     # Drain parser.
├─preprocessing # preprocessing code, data loaders and cutters.
├─representations # Log template and sequence representation.
└─util        # Vocab for DL model and some other common utils.
```

## Datasets

We used `2`  datasets, on python and java2swift respectively. 


| Frameworks | Tensorflow               | PyTorch  |  MXNet | CNTK |
| Tensorflow |           -              |    175   |    152 | 123  |
| PyTorch    | 113                      |  -       |    92  | 67   |


## Reproducibility

We have published an full version of PLELog (including HDFS log dataset, glove word embdding as well as a trained model) in Zenodo, please find the project from the zenodo badge at the beginning.

### Environment

**Note:** 
- We attach great importance to the reproducibility of `PLELog`. Here we list some of the key packages to reproduce our results. However, as discussed in [issue#14](https://github.com/YangLin-George/PLELog/issues/14), please refer to the `requirements.txt` file for package installation.

- According to [issue#16](https://github.com/YangLin-George/PLELog/issues/16), there seems to have some problem with suggested hdbscan version, if your environment has such an error, please refer to the issue for support. Great thanks for this valuable issue!

**Key Packages:**


PyTorch v1.10.1

python v3.8.3

hdbscan v0.8.27

overrides v6.1.0

**scikit-learn v0.24**

tqdm

regex

[Drain3](https://github.com/IBM/Drain3)


hdbscan and overrides are not available while using anaconda, try using pip or:
`conda install -c conda-forge pkg==ver` where `pkg` is the target package and `ver` is the suggested version.

**Please be noted:** Since there are some known issue about joblib, scikit-learn > 0.24 is not supported here. We'll keep watching. 

### Preparation

You need to follow these steps to **completely** run `PLELog`.
- **Step 1:** To run `PLELog` on different log data, create a directory under `datasets` folder **using unique and memorable name**(e.g. HDFS and BGL). `PLELog` will try to find the related files and create logs and results according to this name.
- **Step 2:** Move target log file (plain text, each raw contains one log message) into the folder of step 1.
- **Step 3:** Download `glove.6B.300d.txt` from [Stanford NLP word embeddings](https://nlp.stanford.edu/projects/glove/), and put it under `datasets` folder.
- **Step 4:** Run `approaches/PLELog.py` (make sure it has proper parameters). You can find the details about Drain parser from [IBM](https://github.com/IBM/Drain3).


**Note:** Since log can be very different, here in this repository, we only provide the processing approach of HDFS and BGL w.r.t our experimental setting.


## Anomaly Detection

To those who are interested in applying PLELog on their log data, please refer to `BasicLoader` abstract class in preprocessing/BasicLoader.py` for more instructions.

- **Step 1:** To run `PLELog` on different log data, create a directory under `datasets` folder **using unique and memorable name**(e.g. HDFS and BGL). `PLELog` will try to find the related files and create logs and results according to this name.
- **Step 2:** Move target log file (plain text, each raw contains one log message) into the folder of step 1.
- **Step 3:** Create a new dataloader class implementing `BasicLoader`. 
- **Step 4:** Go to `preprocessing/Preprocess.py` and add your new log data into acceptable variables.

## Contact

We are happy to see `PLELog` being applied in the real world and willing to contribute to the community. Feel free to contact us if you have any question!
Authors information:

| Name          | Email Address          | 
| ------------- | ---------------------- | 
| Lin Yang      | linyang@tju.edu.cn     |
| Junjie Chen * | junjiechen@tju.edu.cn  |
| Weijing Wang  | wangweijing@tju.edu.cn |

\* *corresponding author*

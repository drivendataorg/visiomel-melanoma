[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![Banner Image](https://drivendata-public-assets.s3.amazonaws.com/visiomel_banner_img.jpeg)


# VisioMel Challenge: Predicting Melanoma Relapse

## Goal of the Competition

Melanoma is a cancer of the skin which develops from cells responsible for skin pigmentation. In 2020, over 325,000 people were diagnosed with skin melanoma, with 57,000 deaths in the same year.<sup>[1](https://jamanetwork.com/journals/jamadermatology/article-abstract/2790344#:~:text=Malignant%20melanomas%20(hereafter%20melanoma)%20account,skinned%20populations%20of%20European%20ancestry)</sup> Melanomas represent 10% of all skin cancers and are the most dangerous due to high likelihood of metastasizing (spreading).<sup>[2](https://www.sfpathol.org/actions-data-challenge-2022-23-francais.html)</sup>

Patients are initially diagnosed with melanoma after a pathologist examines a portion of the cancerous tissue under a microscope. At this stage, the pathologist assesses the risk of relapse—a return of cancerous cells after the melanoma has been treated—based on information such as the thickness of the tumor and the presence of an ulceration. Combined with factors such as age, sex, and medical history of the patient, these microscopic observations can help a dermatologist assess the severity of the disease and determine appropriate surgical and medical treatment. Preventative treatments can be administered to patients with high likelihood for relapse. However, these are costly and expose patients to significant drug toxicity.

**Assessing the risk of relapse therefore a vital but difficult task.** It requires specialized training and careful examination of microscopic tissue. Currently, machine learning approaches can help analyze [whole slide images (WSIs)](https://en.wikipedia.org/wiki/Digital_pathology) for basic tasks like measuring area. However, computer vision has also shown some potential in classifying tumor subtypes, and in time may serve as a powerful tool to aid pathologists in making the most accurate diagnosis and prognosis. <sup>[3](https://www.nature.com/articles/s41598-022-24315-1), [4](https://www.sciencedirect.com/science/article/pii/S215335392200743X), [5](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8635121/)</sup>


The goal of this challenge was to predict whether a relapse will occur in the 5 years following the initial diagnosis using digitized versions of microscopic slides.

## What's in this Repository

This repository contains code from winning competitors in the [VisioMel Challenge: Predicting Melanoma Relapse](https://www.drivendata.org/competitions/148/visiomel-melanoma/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

| Place | Team or User                                              | Public Score | Private Score | Summary of Model |
| ---   | ---                                                       | ---          | ---           | --- |
| 1     | [karelds](https://www.drivendata.org/users/karelds/)      | 0.4128 | 0.3940 | Ensemble a number of different models: First, a set of CNN models (SWIN architecture) pretrained with breslow target and finetuned with relapse target. Second, a set of SparseConv models trained using a multiple instance learning approach to predict breslow and ulceration. The breslow and ulceration predictions from the CNN models and relapse prediction from the SparseConv model are provided along with the clinical metadata as features to train a multilayer perceptron to predict relapse. |
| 2     | Tilted Towers [lucas.robinet](https://www.drivendata.org/users/lucas.robinet/) [Kheil-Z](https://www.drivendata.org/users/Kheil-Z/) | 0.4094 | 0.3998 | Train ResNet to predict relapse, ulceration and breslow on low resolution images. Then concatenate the imagery embedding with a clinical embedding, obtained from dense layers. Use a focal loss to handle class imbalance and help the model learn on difficult examples that are often present in WSIs. |
| *    | [marvinler](https://www.drivendata.org/users/marvinler/)   | 0.4081 | 0.4008 | - |
| 3     | [trislaz](https://www.drivendata.org/users/trislaz/)      | 0.3846 | 0.4010 | Train a model to filter out irrelevant patches, then encode with Giga-SSL pre-trained models to obtain embeddings for each WSI. Train logistic regression in a 40-fold cross-validation scheme to predict relapse. |

Additional solution details can be found in the `reports` folder inside the directory for each submission.

*The participant declined to release their code.

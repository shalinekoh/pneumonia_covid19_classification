Development of Deep Learning Model with Visualization for Pneumonia and COVID-19 Prediction from Chest X-Rays
---

**Project summary:** To assist with the global COVID-19 pandemic and various medical diagnosis that involved radiographic techniques, several studies have proposed to detect and classify potential cases by analysing radiological images. X-ray machines are widely available and could potentially provide quick diagnosis for COVID-19 and many other medical conditions. The developed model can easily be adapted to detect problems affecting the bones, joints, internal organs, etc. This work aims to develop an alternative method using image processing on chest X-ray images to provide an efficient and accurate diagnosis. Convolutional neural network models that can detect the presence of COVID-19 and pneumonia infection from chest X-ray images are developed by exploiting transfer learning techniques.  The developed model with the highest performance yielded an accuracy of 98.13%, sensitivity of 97.7%, and specificity of 99.1%. The concept of Gradient Class Activation Map (CAM) was also adopted to highlight the important regions in the X-ray images that leads to the final decision of the trained model. The generated heatmaps are then compared with the annotated X-ray images by board-certified radiologists. Results show that the findings strongly correlate with clinical evidence. 

Table of Content
---
1. [Data Preparation](#id-section2)
2. [Model Development](#id-section3)
3. [Localisation using Grad-CAM](#id-section4)

<div id='id-section2'/>

Data Preparation
---
The dataset used to train the models were obtained from three publicly available sources.
1. [COVID-19 image data collection](https://github.com/ieee8023/covid-chestxray-dataset)
2. [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
3. [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

A total of 9337 images were collected from these sources. 

The collected dataset consists of 795 COVID-19 cases, 2924 normal cases, and 5618 cases of pneumonia. With the COVID-19 class  being  the  minority,  the  dataset’s class  imbalanced problem will affect the model’s accuracy as the model will have the tendency to be bias towards the class with more instances. Hence, random under-sampling techniques are applied to the dataset. This technique involves removing data in a random manner from the majority classes to achieve the same amount of data in the minority class. After removing data from normal  and  pneumonia  classes,  the  dataset  consists  of  795 data  in  each  class. All  of  the collected images were normalised to a common scale and resized as required by the deep learning model. To  enable  the  model  to  generalise  well  to  unseen  data,  data  augmentation  is also implemented on the training set in a random manner. Data augmentations increase diversity in the dataset and could prevent problems like overfitting. Therefore, different data augmentation techniques and values were tested on the dataset to find the best combination that balances overfitting and under-fitting. The data augmentation techniques applied include rotating images at various angle values, zooming, horizontal flipping, and shearing. The dataset is further divided into 2 sets which are the train and test set. The train set consists of 80% of the data, and the remaining fractions for testing purposes. After processing, the combined dataset consists of 1905 images in the train set and 480 in the test set. Each class contains 635 training data and 160 testing data.

<div id='id-section3'/>

Model Development
---
A transfer learning-based approach is employed to train the deep learning model for the detection and classification of COVID-19 and pneumonia in chest X-ray images. The collected dataset only has 2385 images, however, the models used in this paper were pre-trained on over 1 million images from 1000 categories with the ILSVCR database. When retraining with the new COVID-19 dataset that contains only 1905 images, the machine will be able to exploit the previous knowledge gained  from the 1000 categories to improve generalisation and accuracy. To implement this, the model is first instantiated with weights pre-trained on the [ILSVCR database](https://www.researchgate.net/publication/265295439_ImageNet_Large_Scale_Visual_Recognition_Challenge). The model is instantiated without the top classifier layer and replaced with modified custom layers as shown in the figure below. The newly added custom layers consist of 5 layers which are AveragePooling2D layer, Flatten layer, Dense layer with 64 nodes, Dropout layer, and a final Dense layer with 3 nodes and “softmax” activation function for 3 class prediction. During the first 15epochs, the pre-trained weights of the initial convolutional layers are frozen, and only the new modified layers are trained on the new dataset to convert extracted features from the early frozen convolutional layers into 3 class predictions.
Training is then continued for another 15 epochs whereby part of the later layers will be unfrozen. The weights of the later layers are fine-tuned along with the custom layers using the COVID-19 dataset with a lower learning rate. By doing so, significant improvement of the model’s accuracy could be achieved as the pre-trained features will be more adapted to the new dataset. 
In this transfer learning approach, the proposed architecture of the models used are based on:
1. [VGG16](https://arxiv.org/abs/1409.1556)
2. [Inception-ResNet-V2](https://arxiv.org/abs/1602.07261)
3. [Xception](https://arxiv.org/abs/1610.02357)

Model Architecture: (a) VGG16 (b) Inception-ResNet-v2 (c) Xception

![851322ee-a636-4259-b97f-e00f8c538a87](https://user-images.githubusercontent.com/44059891/134813298-339cfa7d-2896-4603-992f-a8da651c6d75.png)

The same methodology is followed by all proposed models whereby the new modified layers of the model are trained on the COVID-19 dataset, and then the last layers of the model is fine-tuned and re-trained along with the new modified layers. The performance of the developed models were evaluated based on their accuracy, specificity, sensitivity, and F1-score. The accuracy and loss of the model for both train and test set were also observed for each epoch, the variation of accuracy and loss during the process of training and testing can be monitored. 

Summarised result: 

<img width="600" alt="Screenshot 2021-09-20 at 5 52 43 PM" src="https://user-images.githubusercontent.com/44059891/134812945-2d3cb668-cac8-484c-8180-e2dd4fc2612d.png">

Accuracy:

<img width="600" alt="Screenshot 2021-09-20 at 5 52 55 PM" src="https://user-images.githubusercontent.com/44059891/134812952-427ec808-5c88-483a-84da-80cf108206ce.png">

As shown in the figure below, the training and testing loss  decrease  with  the  increase  of  training  iteration  epochs  and  gradually  converges  after approximately 20 epochs for the VGG16-based model, and approximately 25 epochs for the Inception-ResNet-v2 based model and Xception-based model. This indicates that the models are learning and no significant overfitting or underfitting problems were observed.

VGG16, Inception-ResNet-v2, Xception:
<img width="850" alt="Screenshot 2021-09-20 at 5 56 05 PM" src="https://user-images.githubusercontent.com/44059891/134812959-f6a6f986-c8e9-455c-9956-ee03d0e1aa81.png">

<div id='id-section4'/>

Localisation using Grad-CAM
---
Gradient-weighted Class Activation Map (Grad-CAM) aims to build trust in intelligent systems by introducing transparency and interpretability in AI models. It  allows intelligent models to explain why the model predicts what they predict. Class activation map techniques are generally utilised to visually debug a deep learning model due to the non-transparent nature of the neural network. However, Grad-CAM was used in the proposed system to localise the potentially infected area to confirm the classification result. Grad-CAM technique can be used in any layer of a network to provide an explanation of the activations. In the proposed system, Grad-CAM is used in the final convolutional layer to explain the decision of the model. To assist the model to localise better, the input chest X-ray images were processed before being read by the proposed model. The lungs region was manually extracted as illustrated below. The extracted lung X-ray image will be taken as input. Once the predicted label has been computed by the proposed model, the Grad-CAM technique is applied on the last convolutional layer to obtain a heatmap that highlights regions with the highest importance. The heatmap generated by the Grad-CAM algorithm will be overlaid on the original image and compared with the radiologist-marked X-ray image.

<img width="500" alt="Screenshot 2021-09-20 at 5 42 03 PM" src="https://user-images.githubusercontent.com/44059891/134812534-da22aa30-8ade-4e59-90c7-98a0c6414ee7.png">

The proposed model that is based on Inception-ResNet-v2 demonstrated promising results at classifying COVID-19, normal, and pneumonia X-rays with a final accuracy of 98.13%. Nevertheless, to improve the reliability of the proposed  model, it is crucial to relate the classification results to clinical evidence. Grad-CAM  algorithm is exploited to compute a heatmap that highlights the highly important regions in the X-ray images that contributed to the final classification result. The heatmap contains intensity values that correspond to the importance of that particular pixel. Higher intensity indicates higher importance to the final result and vice versa. The highlighted region produced by the Grad-CAM algorithm could correspond to the potentially infected region in the lungs. 

![MergedImages](https://user-images.githubusercontent.com/44059891/135054528-ea65c9b9-11ad-45ed-a343-56d69e3253f9.png)

![MergedImages](https://user-images.githubusercontent.com/44059891/135054568-a288de99-a5ae-46ee-a09d-21e616633454.png)

![MergedImages (1)](https://user-images.githubusercontent.com/44059891/135054588-577a9956-a737-48f3-9f4a-6ba84e23da35.png)

![MergedImages (2)](https://user-images.githubusercontent.com/44059891/135054601-b0b84dc1-2233-4dce-aad9-186528589374.png)

![MergedImages (3)](https://user-images.githubusercontent.com/44059891/135054611-bf597791-5272-4c18-9b29-347355b39b4e.png)

The first column shows the original COVID-19 X-ray images that are marked and labeled by a radiologist. The regions that are circled in yellow are the regions that exhibited signs of COVID-19. The second column displays the color-coded heatmap computed using Grad-CAM, whereby the high-intensity regions reflect the area of interest. The third column shows the heatmap that is overlaid on the extracted lung images. The rightmost column contains the pixel intensity of the heatmaps in the second column. The intensity ranges from 0 to 255, the red-colored boxes are the regions that were marked by the radiologist in the original X-ray image. To validate the robustness and credibility of the trained model, a total of 5 COVID-19 chest X-ray images were collected and compared with the Grad-CAM generated heatmap overlaid chest X-ray images. These 5 images exhibit different regions of the infected area to prove that the model inference results are not biased. The region that displayed specific signs of COVID-19 (ROI) were marked by a board-certified radiologist. All 5 images were predicted as COVID-19 by the proposed model. The weights and node values that contributed to these 5 inference results are analysed by the Grad-CAM algorithm and heatmap are computed from these inferences. The heatmap intensity is converted into color-coded by matching yellow bring the highest intensity and blue being the lowest intensity (Column 2). The color-coded heatmaps are then overlaid onto the cropped-out chest X-ray (Column 3). The highlighted regions in the heatmap that have relatively higher intensity values (yellow) mostly  fall  within  the  lungs  region. Even though not all marked regions by the radiologist are highlighted in the heatmap, the region that has the highest intensity value coincides with the region marked by the radiologist. The model is able to identify the affected area in the lungs effectively and make predictions accordingly. This is because the final classification result could be derived based on one problematic area, indicating that to identify all affected areas is not necessary for an accurate COVID-19 diagnosis.
Colorization AI Project for UPC postgraduate course 
Final project for the 2023 Postgraduate course on Artificial Intelligence with Deep Learning, UPC School, authored by Oriol Jorba, Didac Höflich and Xavier Pareras

Advised by Pol Caselles.

# Table of Contents

1. 
1. 
1. 

---
---
## Introduction and Motivation

Colorization is a fascinating area of research in the field of Artificial Intelligence and computer vision that aims to add color to grayscale images automatically. 
By leveraging the power of deep neural networks, colorization models can learn to understand the intricate relationships between the structure of grayscale intensity values and corresponding color spaces. 
The motivation behind colorization has often been to revive historical black-and-white images or recolorize old movies, by applying plausible colors to monochromatic scenes. 

Moreover, colorization has the potential to aid artists, designers, and restoration experts in transforming their creative work. However our main aim is to achive with our limited resources comprehension of diferent architachtures and the effect diferent losses can have on colorization models. 

## Dataset

Our main dataset is CelebA wich is a widely used dataset for research in computer vision and facial recognition. It consists of more than 200,000 celebrity images collected from the internet.
Each image is annotated with 40 attribute labels, providing information about facial characteristics such as age, gender, and presence of accessories like glasses or hats. However for our objective we are only using the images and not the atribute classes or identity.
Celeb A contains images of the size 178×218 however we will downsize them due to computation limitations

[![The CelebA Dataset](./celeb_a.png)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

As we progressed in our project we found necessary to use another Dataset to test one of our hypothesis wich will be explaind in another chapter.
The dataset we choose was Oxford 102 Flowers for fine-grained image classification and plant recognition tasks. Comprising 8,189 images of 102 different flower species. Often used in plant species identification, image segmentation, and visual recognition.
The images are also annotated with the corresponding flower class label but for this project the labels are not going to be used. 

The Dataset is only used to check anhypothesis and thus is not included in our metrics

[![The Oxford102 Dataset](./OX102.jpg)](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)


## The choice of image format

In our case we want to take the images from our Dataset pass them through our model an obtain a result that is plausible. Our Goal is not to reproduce the ground truth but said plausability. 

However choosing in what format the input is when training the model is not of small importance in colorization problems. Normally images are consumed using RGB with its 3 channels, one for each color (red,blue, Green) . However in its article Colorizing black & white images with U-Net and conditional GAN | Towards Data Science Moein Shariatnia states that in most of the “recent” articles he reviewed (2020) notices the RGB model is discarded. 
And L*a*b model is used instead

<img src="./Formats.png"  width=75% height=25%>

Sariatna explains that the problem with training such a model with RGB would be to transform the images from RGB to grayscale and then train the model to predict 3 numbers for each pixel. Instead the literature suggested using the LAB  system. Which already uses a grayscale image that we can use and to rebuild the image and only two other channels. A green-red scale and a blue-yellow scale. Which would make the work of a model substantially easier. 

However taking this format has a series of tradeoffs

### **RGB Format**
  **Advantages**
  - Most Common: RGB is the standard for most images meaning that libraries and many built in functions have this format as its default meaning its simple and straightforward to use. No need to change formats
  - No intrinsic luminosity: This means the model does not need to rely on a luminosity channel as part of its structure its 3 channels build them after they have taken the imput. This forces the model to learn about the structure of the image. 

  **Disadvantages**
  - Harder to predict: having to predict three channels instad of two raises the computational cost as well as the dificulty for the model. 
  -  Perceptual non uniformity: RGB does not conform to the way humans percieve color the Euclidean distance between two colors in the RGB color space does not correspond to their perceived color difference by the human visual system. In other words, a change of the same magnitude in RGB values may result in different perceived changes in color for different regions of the color space.

### **LAB Format**
  **Advantages**
  - Perceptual Uniformity: Conversely to RGB the LAB color space is designed to be perceptually uniform, meaning that the Euclidean distance between colors in the AB channels roughly corresponds to their perceived color difference by the human visual system. In theory this property can lead to more natural and visually appealing colorizations.
  - Reduced complexity: The L channel represents grayscale information, making it unnecessary for the model to predict color for this channel. This simplifies the colorization problem, reducing the complexity of the task and potentially leading to faster convergence during training.

  **Disadvantages**
  - Additional preprocessing: Converting images from RGB to LAB format requires additional preprocessing steps, which may increase the computational overhead during data preparation.
  - Channel dependence: The AB channels' values in LAB format are dependent on the L channel, which means any errors or inaccuracies in the L channel prediction can affect the colorization quality in the AB channels. Moreover the model is not forced to learn the structure of 

We choose LAB format because low computing power and having potentially faster convergence were our main concern.

Similarly to the structure proposed in Influence of Color Spaces for Deep Learning Image Colorization Ballester et al (2022) we feed tha L channel to extract 
predicted ab channels and compare them to the ground truth of the same image according to our criterion. However we do not concatenate them with the L channel or make use of diferentiable RGB function. 

A diagram representing our struture at a high level
<img src="./regular.png"  width=75% height=25%>

## U-NET

As the porpouse of this project was to see the diferences that architechtures and losses have on our output we used a variety of models.


The architecture gets its name from its U-shaped structure, which consists of an encoding path (down-convolution path) and a decoding path (up-convolution path). Let's dive into the details of each part:

Encoding Path:

The encoding path takes the input image and gradually reduces its spatial dimensions while increasing the number of feature channels. It captures high-level abstract representations and context information. The main components of the encoding path are: Down-Convolution: Each down-convolution step consists of a convolutional layer, followed by an activation function such as LeakyReLU, and a normalization layer such as Batch Normalization. These operations help to extract features and introduce non-linearity in the network. Max Pooling: After each down-convolution step, a max pooling operation with a kernel size of 2x2 is performed to reduce the spatial dimensions of the feature maps.

Decoding Path:

The decoding path takes the feature maps from the encoding path and gradually upsamples them back to the original input size while reducing the number of channels. It helps to recover spatial details and refine the segmentation masks. The main components of the decoding path are: Up-Convolution: Each up-convolution step consists of an upsampling operation, often achieved through transposed convolution (also known as deconvolution). This operation increases the spatial dimensions while reducing the number of channels. Concatenation: At each up-convolution step, the feature maps from the corresponding down-convolution step in the encoding path are concatenated with the upsampled feature maps. This skip-connection helps to preserve fine-grained details from the encoding path. Convolution: After concatenation, a regular convolutional layer is applied to refine the combined feature maps. Dropout: In some variations of UNet, dropout layers are introduced to prevent overfitting during training.

Final Layers:

The final layers of the UNet architecture map the refined feature maps to the desired output. Typically, a 1x1 convolution is used to reduce the number of channels to the desired number of output classes (e.g., for semantic segmentation). The activation function used in the final layer depends on the specific task. For example, in binary segmentation, a sigmoid activation is often used, while for multi-class segmentation, a softmax activation is commonly employed. The UNet architecture is symmetric, with the number of channels gradually increasing and then decreasing in the encoding and decoding paths, respectively. This allows the model to capture both local and global information effectively. Additionally, the skip-connections between corresponding layers in the encoding and decoding paths help to bridge the gap between low-level and high-level features, enabling precise segmentation results.

Overall, the UNet architecture has been widely adopted and achieved excellent performance in various image segmentation tasks, making it a popular choice among researchers and practitioners.

<img src="./U-net.png"  width=75% height=25%>

The architecture is implemented using two main classes: UnetBlock and Unet.

The UnetBlock class represents a single block in the UNet architecture. Each block consists of a down-convolution path and an up-convolution path. The down-convolution path reduces the spatial dimensions of the input while increasing the number of channels, and the up-convolution path upsamples the feature maps back to the original size while decreasing the number of channels. Each block also performs normalization and activation operations.

The Unet class represents the entire UNet model. It takes a configuration dictionary as input, which specifies the number of input and output channels, the number of down-sampling steps (n_down), and the number of filters to use in each block (num_filters).

The Unet class initializes the model by creating the innermost block first, which only performs up-convolution. Then, for each down-sampling step (except the innermost and outermost blocks), it creates a new UnetBlock instance with dropout and sets it as the submodule of the previous block. This creates a hierarchical structure where each block's submodule is the previous block.

After the down-sampling steps, the model creates three additional blocks for up-convolution. The number of output filters for each up-convolution block is halved compared to the previous block.

Finally, the outermost block is created with the specified output channels and the submodule set as the entire UNet structure. This block represents the final output of the model.

The forward method of the Unet class passes the input through the entire UNet structure and returns the output

### The results of U-NET

It is important to consider that due to the nature of colorization problems raw validation or test metrics musn't be our guiding factor rather other metrics
**L2 Loss**
Introduction to L2 Distance:
The L2 distance, also known as the Euclidean distance or the Euclidean norm, is a measure of similarity or dissimilarity between two points in a multi-dimensional space. It is widely used in various fields, including machine learning, computer vision, and image processing. The L2 distance calculates the straight-line distance between two points by summing the squared differences of their corresponding coordinates and then taking the square root of the result.

<img src="./L2_Val.png"  width=50% height=25%>

Results from epoch 15(left) and epoch 30(right)

<img src="./L2_15.png"  width=45% height=25%> <img src="./L2_30.png"  width=45% height=25%>


**L1 Loss**

L1 loss, also known as the mean absolute error (MAE), is less sensitive to outliers compared to L2 loss. It calculates the absolute differences between predicted and ground truth colors, which reduces the influence of extreme values. This robustness can be beneficial in colorization tasks where a few deviating pixels should not significantly affect the overall result. Specially if the quality of images is low and they are not clean.

<img src="./L1_val.png"  width=50% height=25%>

Results from epoch 15(left) and epoch 30(right)

<img src="./L1_15.png"  width=45% height=25%> <img src="./L1_30.png"  width=45% height=25%>



**Smooth L1**

Huber loss, also known as the smooth L1 loss, is a loss function that combines the advantages of L1 and L2 losses. It provides a smooth transition between the two by using L2 loss for small errors and L1 loss for larger errors. The Huber loss function depends on a delta that establishes its sensitivity to large errors or outliers

<img src="./Huber_val.png"  width=50% height=25%>

Results from epoch 15(left) and epoch 30(right)

<img src="./Huber_15.png"  width=45% height=25%> <img src="./Huber_30.png"  width=45% height=25%>

As it can be apreciated from the loss and validation of these trials, in a few epochs the model starts to overfit, The model only needs about 10-15 epochs to achieve its maximum methemathical accuracy to the dataset. This is because of the nature of the losses used the "simplicity of the acrhitecture and possibly to the lack of variety in data

Disadvantages of our losses percieved during training

Disadvantages of our losses percieved during training

Lack of Perceptual Awareness: All three losses treat all color differences equally, regardless of their perceptual impact. In colorization tasks, humans tend to perceive certain color deviations more critically than others. For example, a slight shift in skin tone might be more noticeable than a similar shift in a less salient region. L2 loss does not explicitly consider such perceptual differences, which can lead to less visually appealing colorizations.
While this is in theory partly mitigated by our choice of encoding as Lab has perceptual uniformity with.

Limited Color Distribution: L2 loss focuses on minimizing the Euclidean distance between predicted and ground truth colors. However, this may result in over-smoothed colorizations that lack diversity and fail to capture the complexity of real-world color distributions. L2 loss alone may not effectively encourage the generation of rich and diverse color solutions.

Huber Loss: Dependance on hyperparameters, Huber loss depends on an hyperparameter delta that adjustes its sensitivity to outliers thus to make it worthwile trial and error or hyperparameter tunning would be necessary. This might be why perceptually Huber loss seems to be perhaps less realistic applying a simple sepia filter. 

As 


In practice, a combination of different loss functions and regularization techniques is often employed to address the limitations of individual loss functions. Hybrid loss functions that incorporate both L1 and L2 components, or perceptual loss functions based on pre-trained networks, can provide a good balance between preserving details, capturing perceptual differences, and encouraging smooth and visually coherent colorizations.

## Class imbalance in colorization | Testing an Hypothesis

Class imbalance is a known issue with data that can lead to models overpredicting a class due to being overrepresented in the dataset during training. 
It is normally thought as an issue that mainly concerns classification problems or other types of prediction involving classes. 
However it is also present in colorization and by our choice of Dataset we had inadvertently fallen in it. 
The problem being that our dataset CelebA has images with most of its pixels being celebrity faces. The colour tone of skin is usually uniform or very similar even when when seeing diferent skin colors. Even with darker skin specially because our choice to encode with Lab takesout of the account luminosity. 
This leads to algorithms that "lazily" tend to aply the mean distribution of the color instead 

to test this hypothesis we resolved to use a very diferent dataset to run some training epochs and see if that was the case. In this case we choose the Oxford 102 Flower dataset wich is expleined in the introduction. 

The reasoning behind was the diversity of chrominance and distance those colors have from the center of the continous ab distribution


<img src="./Forced_diversity.png"  width=50% height=25%>

After training with the Flowers we can easily see how the results are much more saturated and varied. By having a diferent color distribution simply averaging the pixels does not produce a good result forcing the model to learn the structure of the image and predict acording to it more often. 

<img src="./Flower_E.png"  width=45% height=25%> <img src="./Flower_L.png"  width=45% height=25%>

## Adding a Perceptual Loss

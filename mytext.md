## Abstract

this is my chapter


## 1.Introduction

This project follows the VM1 project. As discussed in VM1, the project was defined based on issues encountered with a platform named bbtwin.This specialized platform was custom-built for B&H by the innovation team. However, the data collected was affected by noise, and the platform contained numerous inaccuracies. Additionally, each measurement had to be manually entered, making the platform impractical for efficient use. To address these challenges, efforts are underway to enhance the platform as part of an innovation project, aiming to significantly improve its functionality and usability.

After encountering these platform issues, a strategy for enhancing the platform involved implementing segmentation. Point cloud segmentation has been a focus of numerous research endeavors,and many articles have been published centered the state-of-the-art in this field of study. Given the B&H dataset, the specific area of operation, and the project's status, the search for a robust pre-trained Neural network model was considered.

Subsequently, discovering a fine-tuned model that utilized an open source dataset from the same domain as the B&H dataset led to the identification of the S3DID dataset, which will be elaborated upon in the subsequent sections.

### Project statment
#### Initial Reaserches
##### Machine Learning

As previously discussed, this project follows the initial approaches of VM1. Numerous initial research studies were conducted during the survey phase. Some of these studies delved into machine learning methods.These methods are adept at handling complex, multi-dimensional datasets such as point clouds. Machine learning algorithms can broadly be classified into supervised, unsupervised, and reinforcement learning, depending on their learning approach.Supervised learning models discern patterns and features within pre-processed datasets to predict class labels or quantities in unknown datasets. Unsupervised learning methods, on the other hand, heuristically detect class information patterns, while reinforcement learning involves learning through trial and error by interacting with the environment.Drawing from this information,supervised machine learning methods are well-suited for classification and object detection in point clouds. Meanwhile, unsupervised machine learning methods prove beneficial for point cloud segmentation.Therefore, the machine learning methods employed in achieving the objectives of point cloud processing steps can be categorized as follows:
* Supervised Classification Methods: Including Deep Learning (DL)-based algorithms, Support Vector Machine (SVM), and Random Forests (RF).
* Unsupervised Clustering Methods: Comprising K-means clustering and hierarchical clustering.  clustering, such as K-means and DBSCAN clustering, which operate based on neighboring data points.[2](https://www.sciencedirect.com/science/article/pii/S1474034621002500) 


In the VM1 DBSCAN was adopted,as a robust unsupervised method for identifying hierarchical patterns. However, applying machine learning techniques to process point clouds for construction-related purposes presents greater challenges compared to other domains. The inherent characteristics of point clouds, including irregularity, unstructured data, and lack of order, pose difficulties for machine learning algorithms to effectively learn from them. Furthermore, the prevalence of occlusion and noise in construction and infrastructure environments significantly impacts the performance of machine learning methods when processing point clouds [1](https://www.sciencedirect.com/science/article/abs/pii/S1474034621002500). Consequently, the VM1 project did not yield optimal results.

According to the information provided in the final paragraph, this project does not utilize any additional machine learning methods beyond those previously mentioned. and Deep learning stands out as a robust approach suitable for handling intricate and layered datasets.

| Method   | Art oF State  |
|-----------------------|-----------|
|SupperviseLearning  |DeepLearnin SVM RandomForest|
| UnSupperviseLearning  |  Clustering |Radius Search |



##### Deep Learning

As previously stated, neural networks or DL based methods, recognized for its ability to handle intricate and layered datasets, involves computational models with multiple processing layers. These models learn and represent data with varying levels of abstraction, mirroring how the human brain comprehends complex information across different modes. This implicit capturing of intricate structures in vast datasets characterizes the essence of deep learning. Within the realm of deep learning lies a diverse range of methods, including neural networks, hierarchical probabilistic models, and various supervised and unsupervised feature learning algorithms.

##### A Brief History

The quest to replicate the human brain's functioning sparked the early development of neural networks. In 1943, McCulloch and Pitts explored how interconnected basic cells, known as neurons, could generate highly complex patterns in the brain. Their model of a neuron, called the MCP model, significantly contributed to the evolution of artificial neural networks.

A pivotal breakthrough in the trajectory of deep learning occurred in 2006 when Hinton et al. introduced the Deep Belief Network. This network featured multiple layers of Restricted Boltzmann Machines, trained greedily in an unsupervised manner, fostering a significant shift in deep architectures and learning algorithms. The guiding principle of training intermediate representation levels using unsupervised learning at each level fueled subsequent developments that propelled the surge of deep architectures and algorithms witnessed over the last decade.[3](Deep Learning for Computer Vision: A Brief Review)

##### Deep Learning For Computer vistion

The collaboration between computer vision and deep learning persistently expands the horizons of what machines can discern and comprehend from visual data. This synergy drives progress across various industries and applications.

As an illustration within the Point Cloud domain as part of computer vision feild, Segmentation involves supervised machine learning techniques categorizing individual points within a point cloud into predefined classes. Deep learning methods, prevalent in construction and infrastructure industries, encompass neural networks, unsupervised and supervised feature learning algorithms, and hierarchical probabilistic models. These methods progressively extract higher-level features from raw input data, elevating representations from a lower to a more abstract level. Although deep learning achieves cutting-edge results akin to human capabilities, it heavily relies on extensive training datasets and computational resources for optimal performance, posing challenges in adopting it for point cloud classification. Deep learning, a subset of machine learning, significantly impacts computer vision and robotics by training deep neural networks—artificial neural networks with multiple layers—to discern patterns in data. This approach is particularly useful for intricate tasks where conventional methods might encounter challenges. Point cloud segmentation, a facet of computer vision, involves processing 3D data and benefits from the capabilities of deep learning, including algorithms like Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and their variants, all showing exceptional performance in various computer vision tasks.

Convolutional Neural Networks (CNNs) are widely used across multiple domains for processing structured data; however, their suitability for unstructured point cloud data is limited due to their dependence on structured grids or voxels. In the subsequent sections of this project, we'll explore PVCNN, a variant like PVConv, which operates on voxels using CNNs. Acquiring a deeper understanding of CNNs can facilitate greater engagement with the project, especially in comprehending the workings of PVCNN.Next, we'll provide a brief overview of CNNs in this section, followed by an exploration of 3D CNNs in the subsequent part.

##### Convolutional neural network

###### CNN Artitect
CNNs, an integral part of neural networks, utilize a mathematical process known as convolution, typically represented by a dot within the network architecture, positioned between the input and output layers.
Convolution, mathematically, is an operation that merges two functions to generate a third function. It essentially amalgamates information from one function with another. Specifically in signal processing or image processing, convolution integrates two functions that characterize a signal, creating a novel function that represents the altered signal.
Consider two sequences of numbers: [2, 1, 0] and [3, 4, 5]. The process of convolving these arrays involves sliding one sequence over the other, conducting element-wise multiplication, and summing the products at each respective position.

To manually compute this convolution:

Given arrays: $$[2, 1, 0]$$ and $$[3, 4, 5]$$
Step 1: Reverse the second array: $$[5, 4, 3]$$

Step 2: Align the reversed array with the first array:
$$[2, 1, 0]$$

$$[5, 4, 3]$$
Step 3: Conduct element-wise multiplication and sum the products:
$$ (2 * 5) + (1 * 4) + (0 * 3) = 10 + 4 + 0 = 14 $$

Convolution stands as a pivotal element in both signal processing and image analysis, exerting its significance by facilitating information extraction, feature accentuation, and crucial operations necessary for comprehensive data understanding and processing. Here's a more formal elucidation of convolution's importance in these domains:
* Feature Extraction: Convolution holds the capacity to extract pertinent information from signals or images. In signal processing, it can discern specific patterns or characteristics existing within the signal, such as distinct frequencies in audio signals. In image analysis, convolutional filters are adept at accentuating edges, textures, or shapes.
* Filtering and Smoothing: Convolution operates as an effective filter, eliminating noise or undesired elements from signals or images. Its application enables smoothing operations, enhancing the clarity of signals or images. For instance, in image processing, convolutional techniques can blur images to reduce noise or sharpen them to delineate edges more distinctly.
* Pattern Recognition: Convolution serves as a robust tool for detecting patterns within signals or images. In image processing, tailored convolutional filters can recognize specific features like lines, curves, or textures. This capability forms the basis for numerous computer vision tasks, including object detection and facial recognition.
* Transformation and Operations: Convolution enables transformational operations on signals or images. In signal processing, it facilitates the conversion of time-domain signals into the frequency domain using suitable kernels (e.g., Fourier Transform). Likewise, in image processing, convolutions can execute tasks such as resizing, scaling, or rotating images.[9](https://math.libretexts.org/Bookshelves/Differential_Equations/Introduction_to_Partial_Differential_Equations_(Herman)/09%3A_Transform_Techniques_in_Physics/9.06%3A_The_Convolution_Operation)

it's time to delve into the neural network and convolution methodology combined. A convolutional neural network (CNN) is structured with layers including an input layer, hidden layers, and an output layer. Within the hidden layers, there are one or more layers dedicated to convolutions. These layers perform dot products between a convolution kernel and the input matrix, often using the Frobenius inner product, with ReLU as the common activation function. As the kernel traverses the input matrix, it produces a feature map, feeding into subsequent layers such as pooling, fully connected, and normalization layers.
In a CNN, the input is represented as a tensor with the shape: (number of inputs) × (input height) × (input width) × (input channels). Upon traversing a convolutional layer, the image transforms into a feature map, often referred to as an activation map, exhibiting the shape: (number of inputs) × (feature map height) × (feature map width) × (feature map channels).

more over, in Convolutional Neural Networks (CNNs), convolutional layers process data similar to neurons in the visual cortex responding to specific stimuli. These layers utilize weight sharing, reducing the need for an excessive number of neurons compared to fully connected networks, especially for large inputs like high-resolution images. This method of convolution helps cut down free parameters, facilitating the creation of deeper networks and addressing gradient problems during backpropagation.

To expedite processing, standard convolutional layers can be substituted with depthwise separable convolutional layers, employing depthwise and pointwise convolutions. Depthwise convolutions execute spatial convolution across individual input tensor channels, followed by pointwise convolutions using 1x1 kernels.

Pooling layers, such as local (e.g., 2x2) and global pooling, consolidate neuron clusters from one layer into a single neuron in the subsequent layer, effectively reducing data dimensions. Max pooling selects the maximum value within local neuron clusters, while average pooling computes the average value.

Fully connected layers establish connections between every neuron in one layer and every neuron in another, resembling a traditional multilayer perceptron neural network. These layers handle flattened matrix data to classify images.

The receptive field in CNNs refers to the area from which each neuron in a layer gathers input. Convolutional layers restrict this field to a defined region, while fully connected layers consider the entire preceding layer. Modifying the receptive field size in a CNN can be achieved using methods like atrous or dilated convolution, providing flexibility without a surge in the parameter count.

Weights in neural networks are vectors utilized to compute neuron output values from input data. In CNNs, filters signify specific input features, and weight sharing enables multiple neurons to use the same filter. This approach conserves memory by allowing shared biases and weight vectors across receptive fields that share a filter.[8](https://en.wikipedia.org/wiki/Convolutional_neural_network)

###### The superiority of CNN
Convolutional Neural Networks (CNNs) has some superority to other methods, particularly in their approach to handling data and their architecture.In the following, a brief overview of some of these methods is presented.
* Feature Learning:CNNs autonomously learn hierarchical features from raw data, reducing the need for manual feature extraction compared to traditional machine learning models.
* Local Connectivity: CNNs leverage local connectivity to process data. Neurons are connected to small, local regions in the input, allowing the network to learn spatial hierarchies.
* Weight Sharing: CNNs use shared weights and biases across the network, reducing the number of parameters and enabling the model to generalize better to new data.
* Convolutional Layers: These layers apply convolution operations to input data, extracting spatial hierarchies and features.
* Pooling Layers: These layers help in down-sampling the extracted features, reducing computational complexity and memory requirements.
* Hierarchical Structure: CNNs have a hierarchical structure with multiple layers, such as convolutional, pooling, and fully connected layers, enabling them to learn intricate patterns and features.
* Suitability for Image Data: CNNs are particularly well-suited for tasks like image recognition and computer vision due to their ability to capture spatial relationships in data efficiently.
Convolutional neural networks (CNNs) were initially developed to process 2D images. The emergence of 3D CNNs served as an extension to handle volumetric data and spatiotemporal information in 3D space. In the upcoming section, we'll delve into the realm of 3D CNNs.

##### 3D Deep Leraning
As previously discussed, point clouds belong to the realm of computer vision but in a 3D format. Expanding on the role of deep learning in computer vision, we encounter a specialized branch known as 3D deep learning. This branch specifically applies deep learning techniques crafted for the processing and analysis of three-dimensional data. While traditional deep learning has primarily concentrated on handling two-dimensional data such as images and videos, 3D deep learning broadens these capabilities to manage volumetric data or information represented in three-dimensional space. This specialized field tackles diverse challenges related to comprehending and extracting meaningful insights from 3D data, encompassing tasks like handling point clouds, analyzing volumetric scans (such as CT or MRI scans), reconstructing 3D representations from multiple perspectives, and interpreting spatial-temporal information present in videos.

Several techniques and architectures have been developed in 3D deep learning to handle these data types effectively. These include:
* 3D Convolutional Neural Networks (CNNs): These networks are designed to process and learn from volumetric data by extending the concept of 2D convolutions to the third dimension, enabling them to capture spatial information in 3D space.
* PointNet and PointNet++: These architectures are tailored for processing unstructured point cloud data directly, allowing for tasks like segmentation, classification, and object recognition without converting the data to regular grids or voxels.
* Volumetric Representations:Methods that convert 3D data into volumetric grids (voxels) for processing using traditional 3D CNNs.
* Graph Neural Networks (GNNs):GNNs are employed for irregular or graph-structured 3D data, like molecular structures or social networks in 3D space.

Exploring and analyzing point clouds as three-dimensional data guides us toward delving further into 3D analysis. Conducting a comparative study between 2D and 3D analysis can significantly enhance our understanding of the intricacies within 3D analysis. Various fields such as medicine, construction, robotics, virtual reality, and autonomous vehicles extensively deal with diverse forms of 3D data. Employing techniques like object detection and segmentation in these fields necessitates a deeper exploration of 3D analysis.

Scrutinizing the fundamental disparities between 3D data analysis and its 2D counterpart offers valuable insights into comprehending 3D analysis more comprehensively. This understanding is rooted in the distinct nature of the data being processed and the methodologies employed for its interpretation and manipulation. As a result, delving into the nuances of 3D analysis and its methodologies lays the groundwork for advancements in 3D deep learning techniques.

##### 3D data analysis vs 2D data analysis

The subsequent points briefly outline fundamental disparities between 3D and 2D data analysis.
* Dimensionality:
* * 2D Data:
* * * Representation: Two-dimensional data is visualized on a flat plane, typically with two axes: width (X-axis) and height (Y-axis).
* * * Examples: Images, videos, documents, and surfaces are typical examples of 2D data. In images, each pixel represents color or intensity values at specific X and Y coordinates.
Information: 2D data lacks depth information, so it primarily captures surface details without any volumetric or spatial depth perception.
* * 3D Data:
* * * Representation: Three-dimensional data exists in a spatial environment with three axes: length (X-axis), width (Y-axis), and depth (Z-axis).
* * * Volumetric scans like CT (Computed Tomography) or MRI (Magnetic Resonance Imaging), point clouds generated from LiDAR scans, 3D models, and spatial-temporal data in videos are instances of 3D data.
* * * Information: 3D data includes volumetric information and spatial depth. For instance, in a CT scan, each voxel (3D pixel) not only contains X and Y positions but also depth information, representing structures within the body in a spatial context.
* Complexity and Spatial Relationships:
* * 2D Data:
* * * Representation: 2D data primarily deals with information on a two-dimensional plane and often represents surfaces or visual data (e.g., images, documents, or graphics).
* * * Spatial Relationships: While 2D data captures spatial relationships along the X and Y axes (width and height), it lacks information concerning depth or volumetric aspects.
* * * Example: In an image, the arrangement of pixels provides spatial information about how elements relate within the flat surface, such as object positions, edges, or patterns, without considering their depth or volumetric properties.
* * 3D Data:
* * * Representation: Three-dimensional data encompasses spatial relationships along the X, Y, and Z axes, providing depth, volume, and shape information.
* * * Spatial Relationships: 3D data captures intricate spatial relationships in three dimensions, including not only width and height but also depth. This includes volumetric details, object shapes, relative distances, and the spatial layout within the three-dimensional space.
* * * Example: In medical imaging, a 3D MRI scan of the brain not only reveals the structures' surface details (2D representation) but also depicts their volumetric properties and spatial relationships within the brain, aiding in diagnosing complex conditions like tumors by examining their 3D shapes and locations.
To enhance clarity on this subject, we have provided several illustrative examples.
* * * * 2D Image: An image of a car provides a flat representation showing the car's outline and features but lacks depth perception or information regarding its three-dimensional structure.
* * * * 3D Model: A 3D CAD model of the same car provides a comprehensive three-dimensional representation, capturing not only the car's external appearance but also its depth, volume, and spatial relationships. This enables detailed analysis, such as simulating its aerodynamics or evaluating interior space.as a result while 2D data focuses on surface information and limited spatial relationships, 3D data offers a more comprehensive understanding by incorporating volumetric details and intricate spatial relationships within a three-dimensional environment.
* Network Architecture:
* * 2D Data:
* * * CNNs for 2D Data: Convolutional Neural Networks (CNNs) are extensively employed for analyzing 2D data such as images or videos.
* * * Components: Traditional CNN architectures for 2D data include convolutional layers, pooling layers (such as max pooling or average pooling), and fully connected layers at the end of the network.
* * * Functionality: Convolutional layers perform feature extraction by sliding kernels across two spatial dimensions (width and height) to capture patterns and features within the data. Subsequent pooling layers reduce spatial dimensions, and fully connected layers process the extracted features for classification or regression tasks.
* * * Example: In image classification, a 2D CNN may consist of convolutional layers to extract visual features (e.g., edges, textures), pooling layers for downsampling, and fully connected layers for classification.
* * 3D Data:
* * * 3D CNNs for 3D Data: Designed specifically for handling three-dimensional data, 3D CNNs are tailored to process volumetric information.
* * * Components: 3D CNN architectures integrate 3D convolutional layers that operate across three spatial dimensions (length, width, and depth).
* * * Functionality: 3D convolutional layers convolve across the volumetric data, capturing spatial features and patterns in three dimensions. This allows them to comprehend spatial relationships, depth information, and volumetric details present in 3D data.
* * * Example: In medical imaging, a 3D CNN can process volumetric scans (e.g., CT or MRI scans) by utilizing 3D convolutions to capture spatial information in three dimensions, aiding in tasks like tumor segmentation or disease diagnosis.
* Training Challenges:
* * 2D Data:
* * * Computational Requirements: Training models on 2D data generally demands fewer computational resources compared to 3D data due to lower dimensionality.
* * * Data Size: Models analyzing 2D data often require smaller datasets, as they operate in two spatial dimensions.
* * * Processing Complexity: With fewer dimensions, computations within 2D models are relatively less complex compared to 3D, requiring less computational power.
* * * Example: Image classification tasks using 2D CNNs may involve processing images (e.g., 256x256 pixels) that demand moderate computational resources for training.
* 3D Data:
* * Higher Computational Demands: Analyzing 3D data demands more computational power due to the increased complexity arising from the additional spatial dimension.
* * Larger Datasets: 3D models often require larger datasets to effectively capture volumetric information and spatial relationships.
* * Complex Algorithms: Processing three-dimensional structures involves more sophisticated algorithms to comprehend volumetric details, spatial relationships, and depth information.
* * Example: Training a 3D CNN for medical imaging tasks using volumetric scans (e.g., CT scans with hundreds of slices) demands substantial computational resources for processing volumetric data.
* Applications of 2D Data Analysis:
* * Image Classification: Assigning predefined categories to images based on their visual content. Examples include identifying objects, scenes, or classifying images into distinct categories (e.g., classifying animals in wildlife images).
* * Object Detection: Locating and classifying objects within images. This involves identifying multiple objects of interest and drawing bounding boxes around them (e.g., detecting pedestrians or vehicles in autonomous driving scenarios).
* * Facial Recognition: Identifying and verifying individuals by analyzing facial features and patterns. It's widely used in security systems, access control, and identity verification applications.
* * Scene Understanding: Analyzing and interpreting complex scenes in images, which involves identifying objects, their relationships, and contextual understanding (e.g., understanding indoor scenes or outdoor landscapes).
* Applications of 3D Data Analysis:
* * Medical Imaging: Utilizing 3D data analysis, such as CT scans and MRI images, for diagnosing diseases, detecting abnormalities, surgical planning, and treatment monitoring in medical fields
* * Robotics: Incorporating 3D data analysis in robotics for tasks like simultaneous localization and mapping (SLAM), object manipulation, path planning, and 3D perception for robot navigation.
* * Architectural and Construction Industries: Implementing point cloud data for creating as-built models, building information modeling (BIM), renovation planning, clash detection, and quality control in construction projects.


During our exploration of the differences between 3D and 2D data analysis, we introduced some novel concepts that hold significant importance in 3D analysis. We will now offer brief yet informative definitions for each of these concepts.
##### spatial features
Spatial features in 3D refer to the characteristics or attributes present within a three-dimensional space that provide information about the arrangement, relationships, and geometry of objects or elements within that space. These features capture the structural properties and layout of entities in three dimensions.
Some examples of spatial features in 3D data include:
* Shape and Structure: Spatial features encompass the shape, size, and geometry of objects or entities present in a three-dimensional space. This includes information about their contours, surfaces, volumes, and overall structural composition.
* Orientation and Position: Features related to the orientation, position, and alignment of objects in 3D space provide critical spatial information. This includes coordinates, angles, distances, and relative positions of various elements.

* Depth and Distance: Spatial features in 3D encompass depth-related information, such as distance from a reference point, depth perception, or depth changes within the space. This information helps understand the relative positions of objects along the depth axis.
* Spatial Relationships: These features describe how different objects or components relate to each other in three-dimensional space. It includes aspects like proximity, adjacency, containment, intersection, and spatial arrangements between entities.
* Spatial Context: Contextual information within the 3D space, such as the surrounding environment, contextual relationships, and spatial context between objects, contributes to spatial features.
#### State-of-the-Art in 3D Deep Learning
##### 3D Deep Learning 

In the previous conversation, a brief overview of 3D data analysis was provided, emphasizing crucial aspects. In this section, we will delve deeper into 3D deep learning techniques, specifically targeting volumetric analysis, spatial analysis, and point cloud analysis. Let's proceed by exploring the key components and methodologies linked with these techniques.
##### Specialized Architectures:
As previously discussed, similar to the diversity seen in 2D deep learning, 3D deep learning also encompasses various specialized architectures tailored explicitly for efficient processing and analysis of three-dimensional data structures. These architectures are specifically developed to tackle the complexities inherent in volumetric, spatial-temporal, or point cloud data. Some notable architectures include CNNs, RNNs, GNNs, Hybrid Architectures, and models employing Attention Mechanisms. In the subsequent sections, we will conduct an in-depth exploration focusing on CNNs in relation to the project and model utilized for segmentation tasks.
* 3D Convolutional Networks (3D CNNs): 
3D Convolutional Neural Networks (CNNs) are neural network architectures specifically designed to process and analyze three-dimensional data. While traditional CNNs excel in handling two-dimensional data like images, 3D CNNs extend this capability to volumetric data or data represented in three-dimensional space.
These networks employ 3D convolutional layers that operate across three spatial dimensions (length, width, and depth) to capture spatial information within volumetric data. Similar to 2D CNNs, 3D CNNs use filters/kernels to perform convolutions, extracting features from the input volume by sliding the filters through the three-dimensional space.
3D CNNs have applications in various domains, including medical imaging (analyzing CT or MRI scans), video analysis (recognizing actions or activities in videos), 3D object recognition, and computational biology (analyzing molecular structures), among others. They are particularly useful when working with data that has spatial relationships and temporal dependencies across three dimensions.

###### presenting a survey of the techniques and architectures of 3D CNNs

The 3D Convolutional Neural Network (3D CNN) stands as a sophisticated extension of Convolutional Neural Networks (CNNs), specifically tailored to analyze and comprehend three-dimensional data structures. Unlike traditional CNNs that process two-dimensional data, the 3D CNN integrates temporal dimensions, enabling the exploration of spatial and temporal intricacies within volumetric data. Its architectural framework encompasses various layers including input, convolutional, activation, pooling, and a classification layer.

However, Within the 3D CNN, the convolutional layers act as fundamental units extracting intricate spatial and temporal features inherent in the volumetric input data. Activation functions introduce non-linear elements essential for modeling complex relationships within the extracted features. Feature maps, representing discerned features like edges or textures, are constructed through convolutional filters applied to the input data. Moreover, the process of max-pooling reduces the feature map size, ensuring the retention of critical features while optimizing computational efficiency.

furthermore, It is imperative to note this here ,the network's classification layer, primarily comprised of fully connected layers, leverages the learned features to generate predictions based on recognized elements like edges, textures, or shapes. Mathematically, the convolutional process involves the transformation and amalgamation of information from input signals utilizing convolutional operators, allowing the network to discern intricate patterns and structures within three-dimensional data.[4](https://www.neuralconcept.com/post/3d-convolutional-neural-network-a-guide-for-engineers)

Below is a concise overview outlining the architecture commonly found in a standard 3D Convolutional Neural Network (3D CNN).
* Input Layer:The input to a 3D CNN consists of three-dimensional data, often represented in the form of volumetric tensors. These tensors encapsulate the spatial and temporal information of the input, such as videos or volumetric medical imaging data.
* Convolutional Layers:These layers are the core components of the network, performing three-dimensional convolutions. The 3D convolutional operation involves sliding a three-dimensional filter (also known as a kernel) across the input data volume to extract spatial features across width, height, and depth dimensions. These layers learn and extract hierarchical spatial and temporal features from the input data.
* Activation Functions:Activation functions introduce non-linearities into the network, allowing it to model complex relationships within the extracted features. Common activation functions include ReLU (Rectified Linear Unit), Leaky ReLU, or others suitable for capturing non-linearities in three-dimensional data.
* Pooling Layers: Pooling layers reduce the spatial dimensions of the feature maps generated by the convolutional layers. Max-pooling or average-pooling operations are applied across the width, height, and depth of the feature maps, preserving essential features while downsampling the data, which helps in reducing computation and controlling overfitting.
* Fully Connected Layers: After the convolutional and pooling layers, fully connected layers are employed to process the learned features. These layers consolidate the spatial-temporal information extracted from the previous layers and map it to the output classes or categories.
* Output Layer:The final layer in the network produces the output based on the learned representations. For classification tasks, this layer typically uses a softmax function to generate probabilities for different classes. For regression tasks, it might output continuous values.[6](https://www.sciencedirect.com/science/article/pii/S0927025620303414#f0015)

![an example from 3D CNNs]("C:/Users/taghotbi/Desktop/VM2_Writting/1-s2.0-S0927025620303414-ga1_lrg.jpg")

#### Pointcloud as a 3D data

3D Convolutional Neural Networks (CNNs) are recognized for their efficacy in analyzing 3D data; however, they might not be the most suitable option for analyzing point clouds as indivial points, due to specific limitations. These constraints arise from the challenges posed by unstructured and irregular data representations within point clouds. Unlike the structured grid pattern found in typical CNN applications with image pixels, point clouds lack this predefined grid structure. The irregular nature of point cloud data presents difficulties for CNNs in efficient feature extraction and processing without converting points into structured forms like voxels. This conversion process leads to escalated computational complexity and potential information loss.

To address these limitations, PointNet, a specialized neural network architecture, was introduced specifically tailored for point cloud data processing. PointNet operates directly on raw point cloud data without the need for prior voxelization or grid-based representation. It uses a unique set abstraction and transformation layers that handle the unordered nature of point clouds, enabling the network to learn directly from the raw point coordinates.[7](https://www.mdpi.com/1424-8220/21/16/5574)

| Method     | Sampling  | Grouping  |  Mapping  | DataSet  |
|------------|-----------|-----------|-----------|----------|
|Pointnet    |           |           |   MLP     | S3DIS SahpeNet ModelNet40 | 
|Pointnet++  |  Farthest Point Sampling |Radius Search |MLP| S3DIS SahpeNet ModelNet40 |
|PointCNN|Random Sampling|KNN| MLP|S3DIS SahpeNet ModelNet40 | 
|DGCNN| |KNN| MLP|S3DIS SahpeNet ModelNet40 |
|PointConv|Uniform Sampling|Radius Search| MLP|S3DIS SahpeNet ModelNet40 |
|PVCNN|Uniform Sampling|Radius Search|MLP| S3DIS SahpeNet ModelNet40 |
|SpiderCNN|Uniform Sampling| KNN|Tazlor Expansion|S3DIS SahpeNet ModelNet40 |
|Kd-Net| |Tree Baes Nodes|Affine Transformaitions | S3DIS SahpeNet ModelNet40 |
|SO-Net|SOM Nodes|Radius Search|MLP| S3DIS SahpeNet ModelNet40 |
|A-CNN|Uniform Sampling|KNN|MLP| S3DIS SahpeNet ModelNet40 |
|RS-CNN|Uniform Sampling|Radius Search|MLP| S3DIS SahpeNet ModelNet40 |

#### PoinNet

Prior to delving deeper, it is necessary to introduce a range of topics related to neural networks

##### feature alignment

###### Bat Algorithm

The Bat algorithm is a population-based metaheuristics algorithm designed to address continuous optimization problems. Its applications span across optimizing solutions in various fields such as cloud computing, feature selection, image processing, and control engineering problems.
The concept of Bat algorithms originates from the idealized echolocation traits of microbats. These algorithms are developed based on certain simplified rules:
* All bats utilize echolocation to sense distance and can distinguish between food/prey and background barriers.
* Bats navigate randomly at a specific position and velocity while emitting pulses at varied frequencies and wavelengths to locate prey. They adjust these emitted pulses based on the target's proximity.
* Loudness varies from a significant value to a constant minimum, though there are numerous possible variations.
This model doesn't employ ray tracing to estimate time delay or three-dimensional topography, which could be advantageous in computational geometry but is computationally intensive for multidimensional cases.

Additionally, frequency (f) within a certain range corresponds to a range of wavelengths (λ). For example, a frequency range of [20kHz, 500kHz] corresponds to wavelengths from 0.7mm to 17mm.

For implementation ease, any wavelength within a detectable range can be chosen. Adjusting the wavelengths allows altering the range, ensuring it matches the domain of interest and gradually narrowing it down. Moreover, frequency variation while maintaining a fixed wavelength is viable due to the constant product of wavelength and frequency.

In these algorithms, higher frequencies indicate shorter wavelengths, suitable for distances typically covered by bats, which are within a few meters. The pulse rate ranges from 0 (no pulses emitted) to 1 (maximum pulse emission rate).

And the algorithm is : 
```Pseudo
(Objective function f(x), x = (x1, ..., xd)T
Initialize the bat population xi (i = 1, 2, ..., n) and vi
Define pulse frequency fi at xi
Initialize pulse rates ri and the loudness Ai
while (t <Max number of iterations)
Generate new solutions by adjusting frequency,
and updating velocities and locations/solutions [equations (2) to (4)]
if (rand > ri)
Select a solution among the best solutions
Generate a local solution around the selected best solution
end if
Generate a new solution by flying randomly
if (rand < Ai & f(xi) < f(x∗))
Accept the new solutions
Increase ri and reduce Ai
end if
Rank the bats and find the current best x∗
end while
Postprocess results and visualizatio)
```
In simulations, virtual bats are inherently utilized. It's essential to establish the guidelines governing the updates of their positions $$ xi $$ and velocities $$ vi $$ within a search space of d dimensions. The updated solutions $$ xti $$ and velocities $$ v
t
i $$ are determined by
$$ fi = fmin + (fmax − fmin)β, $$
$$ v
t
i = v
t−1
i + (x
t
i − x∗)fi
, $$
$$ x
t
i = x
t−1
i + v
t
i
, $$
The variable β, ranging between 0 and 1, represents a randomly generated vector from a uniform distribution. In this context, x∗ denotes the current globally best solution, determined after comparing all solutions among n bats.
![Bat Algorithm]("C:/Users/taghotbi/Desktop/VM2_Writting/figure-bat-algorithm-1-e1678289097406.jpg")

The product λi * fi signifies the velocity increment. The adjustment of velocity change is achieved by utilizing either fi or λi while keeping the other factor fixed, based on the nature of the problem at hand. In our approach, we set fmin to 0 and fmax to 100, considering the size of the problem domain. Initially, each bat is assigned a frequency randomly chosen from the range $$ [fmin, fmax] $$[11](Nature-Inspired Optimization Algorithms, Chapter 11 - Bat Algorithms)

Regarding the local search process, after selecting a solution among the current best solutions, a new solution for each bat is locally generated using a random walk approach.$$ xnew = xold + ǫAt
,$$
where ǫ ∈ [−1, 1] is a random number, while $$ At =<At
i > $$ is the average loudness of all the bats at this time step.The process of updating the velocities and positions of bats shares similarities with the approach used in standard particle swarm optimization. Here, the parameter fi plays a crucial role in governing the speed and extent of movement among the swarming particles. One can perceive BA (Bat Algorithm) as a harmonious blend of standard particle swarm optimization and an intense local search, regulated by factors like loudness and pulse rate.[10](https://arxiv.org/pdf/1004.4170.pdf)

###### Unsupervised domain adaptation
Unsupervised domain adaptation (UDA) stands as a pivotal technique in machine learning, addressing the common challenge of domain shift between training and deployment data. In many real-world applications, models trained on data from one domain struggle to perform well when faced with data from a different but related domain. UDA, however, seeks to bridge this gap by leveraging knowledge from a labeled source domain and applying it to an unlabeled target domain, thereby enhancing a model's performance in the target domain.

The fundamental premise of UDA lies in its ability to generalize from a source domain, where labeled data is available, to a target domain, where labeled data is scarce or unavailable. The crux of this adaptation process involves mitigating the distributional mismatch or discrepancy between these domains. This discrepancy arises due to variations in data distribution, such as differences in statistical properties, feature spaces, or environmental factors, thereby hindering direct application of models trained on one domain to another.

The workflow of UDA begins by training a model on labeled data from the source domain to learn representations that encapsulate relevant features for the intended task. However, transferring this learned knowledge directly to the target domain poses a challenge due to the domain shift. To address this, UDA employs strategies to align or adapt the source and target domains, minimizing their distributional differences.

Various techniques are employed within UDA frameworks to achieve domain adaptation. One prominent approach involves domain adversarial training, where a domain discriminator is incorporated into the model architecture. This discriminator aims to distinguish between source and target domain data, encouraging the model to learn domain-invariant representations that capture the shared characteristics across domains.

Another strategy involves self-training or pseudo-labeling, leveraging the model's predictions on unlabeled target domain data to iteratively improve its performance. By assigning pseudo-labels to unlabeled target domain samples based on the model's confident predictions, these samples are then used as if they were labeled data for further training iterations, promoting domain adaptation.

UDA finds extensive applications across diverse domains, including computer vision, natural language processing, and speech recognition. In computer vision, for instance, transferring knowledge from synthetic or labeled datasets to real-world or unlabeled datasets enhances the robustness and adaptability of models for tasks like object detection, image classification, or semantic segmentation.[11](Handbook of Statistics, Chapter 5 - Source distribution weighted multisource domain adaptation without access to source data,Sk Miraj Ahmed, Dripta S. Raychaudhuri, Samet Oymak, Amit K. Roy-Chowdhury )





























#### Statment 2

### Background Information 

##### Support A
##### Support B


##### Support A
##### Support B

#### Statment 3

##### Support A
##### Support B


## 2.Body


## 3.Conclusion


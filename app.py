!pip install opencv-python
import streamlit as st
from PIL import Image, ImageEnhance

@st.cache
def load_image(img):
  im = Image.open(img)
  return im

st.set_page_config(page_title='Face Mask Detection', page_icon=None, layout="centered", 
                   initial_sidebar_state="auto")
def main():
  # Menu
  menu = ["Home", "About", "Notebook", "Run it Yourself - Image", "Run it Yourself - Webcam"]
  choice = st.sidebar.selectbox('Navigation Bar', menu)

  # Home Page
  def home():
    # Streamlit title for my ML Project
    st.title("Face Mask Detection using Deep Learning")

    # Project Description
    st.subheader("Project Description")

    # Introduction
    st.write("Hello, my name is Sriram Muthu and I am the creator of this Streamlit App which displays my project, Face Mask Detection with Python using MobileNetV2. And to indulge more into this topic, I have utilized the power of Tensorflow and it's API, Keras, to build a Deep Learning Model that recognizes people with masks on and off.")
    st.write("The novel COVID-19 virus has forced us all to rethink how we live our everyday lives while keeping ourselves and others safe. Face masks have emerged as a simple and effective strategy for reducing the virus’s threat and also, application of face mask detection system are now in high demand for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety. Therefore, the goal of the today’s article is to develop a face mask detector using deep learning.")

    st.subheader("Here\'s an example")
    st.write('In this Face Mask Detection Streamlit App now I will take you through the Coding/Programming parts, but first take a look at an example of the prediction before you go any further.')
    image = Image.open('/content/Screen Shot 2022-06-10 at 3.49.05 PM.png')
    st.image(image, caption='This image above is a detected mask image.')

    st.write("Above you can see the selected person in the Unsplash image is wearing a mask therefore the model predicts and chooses correctly, as the model had a validation accuracy of exactly 99.89.")

    # Table of Contents
    st.markdown(
      """
      <style>
      .header4 {
        padding-top: 20px !important;
      }

      .undordered {
        padding-bottom: 20px !important;
      }
      </style>
      """,
      unsafe_allow_html=True
    )

    st.markdown(
      f"""
      <h4 class="header4">Table of Contents</h4>

      <ul class="unordered">
        <li>About the Dataset</li>
        <li>MobileNetV2 (CNN) Architecture</li>
        <li>Training/Evaluation and EDA</li>
        <li>Experiments and Results</li>
      </ul>
      """,
      unsafe_allow_html=True
    )

    # About the Dataset
    st.subheader("About the Dataset")
    st.write("The images used in the dataset are real images of people wearing mask i.e. the dataset doesn’t contain morphed masked images. The dataset consists of 3835 images belonging to two classes: ")
    st.markdown(
    f"""
    <ul class="unordered2">
      <li>with_mask: 1916 images</li>
      <li>without_mask: 1919 images</li>
    </ul>
    """,
    unsafe_allow_html=True
    )

    st.write("The images were collected from the following source:")
    st.markdown(
    f"""
    <ul class="unordered3">
      <li>Kaggle datasets (Specifically face mask 12k images dataset)</li>
    </ul>
    """,
    unsafe_allow_html=True
    )

    # Technical Parts
    st.subheader('MobileNetV2 Network')
    st.write("MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an inverted residual structure where the residual connections are between the bottleneck layers.")

    image2 = Image.open('Screen_Shot_2020-06-06_at_10.37.14_PM (1).png')
    st.image(image2, caption='Plotted version of MobileNetV2 Layers.')

    st.markdown("##### What is a CNN?")
    st.write('The convolutional neural network or convnets is a major back through in the field of deep learning. CNN is a kind of neural network, and they are widely used for image recognition and classification., They are mainly used for identifying patterns in the image. We don’t feed features into it, they identify features by themselves. The main operations of CNN are Convolution, Pooling or Sub Sampling, Non-Linearity, and Classification.')

    # If you dont know...
    st.markdown('##### If You Didn\'t know...')
    st.write('Deep learning is a type of machine learning and artificial intelligence (AI) that imitates the way humans gain certain types of knowledge. Deep learning is an important element of data science, which includes statistics and predictive modeling.')
    st.write('Computer vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs — and take actions or make recommendations based on that information. If AI enables computers to think, computer vision enables them to see, observe and understand. Computer vision works much the same as human vision, except humans have a head start. Human sight has the advantage of lifetimes of context to train how to tell objects apart, how far away they are, whether they are moving and whether there is something wrong in an image. Computer vision trains machines to perform these functions, but it has to do it in much less time with cameras, data and algorithms rather than retinas, optic nerves and a visual cortex. Because a system trained to inspect products or watch a production asset can analyze thousands of products or processes a minute, noticing imperceptible defects or issues, it can quickly surpass human capabilities.')

    # Training and EDA (Exploratory Data Analysis)
    st.subheader("EDA and Training Code Walkthrough")
    st.markdown("#### The EDA...")

    # EDA
    code3 = """
    import numpy as np # linear algebra
  import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
  import matplotlib.pyplot as plt
  import seaborn as sns
  import tensorflow as tf
  from tensorflow.keras import models
  from tensorflow.keras import layers
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras.preprocessing.image import load_img
  import cv2
  import random

  import os
  i = 0
  for dirname, _, filenames in os.walk('/kaggle/input'):
      for filename in filenames:
          i += 1
          if i < 10:
              print(os.path.join(dirname, filename))
    """
    st.code(code3, language='python')
    st.write("This is where I import all of my libraries that I use in this project. and in the second half of the cell I check the input directories for the data and check file names as references.")

    code4 = """
    print(os.listdir("/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset"))
  print(os.listdir("/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/Train/"))

  print('Number of training images in WithMask section:', len(os.listdir((train_dir + 'WithMask'))))
  print('Number of training images in WithoutMask section:', len(os.listdir((train_dir + 'WithoutMask'))))

  print('Number of validation images in WithMask section:', len(os.listdir((validation_dir + 'WithMask'))))
  print('Number of validation images in WithoutMask section:', len(os.listdir((validation_dir + 'WithoutMask'))))

  print('Number of test images in WithMask section:', len(os.listdir((test_dir + 'WithMask'))))
  print('Number of test images in WithoutMask section:', len(os.listdir((test_dir + 'WithoutMask'))))
    """

    st.code(code4, language='python')
    st.write("This projet is not a typical Data Analysis project so the EDA section isnt to impressive but mainly in this code cell I try to get information about the different data directories.")

    import base64 
    code = '''def show_withmask(num_imgs):
      plt.figure(figsize=(12, 7))
      for i in range(0, num_imgs):
          rand = random.choice(os.listdir(train_dir + 'WithMask'))
          plt.subplot(1, 5, i+1)
          img = load_img(train_dir + 'WithMask/' + rand)
          plt.xlabel('WithMask')
          plt.imshow(img)
      plt.show()
      
  show_withmask(5)'''

    st.code(code, language='python')

    IMAGE = "Screen Shot 2022-06-10 at 5.57.24 PM.png"
    st.markdown(
      """
      <style>
      .container {
          display: flex;
      }
      .logo-text {
          font-weight: 100 !important;
          font-size: 15px !important;
          padding-left: 25px !important;
          padding-top: 20px !important;
      }
      .logo-img {
          float: right;
          padding-bottom: 50px !important;
      }
      </style>
      """,
      unsafe_allow_html=True
    )

    st.markdown(
      f"""
      <div class="container">
          <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(IMAGE, "rb").read()).decode()}">
          <p class="logo-text">As you can see to the left there is the code cell's output, which is a training image used to develop my Deep Learning Model, although this is just one in the vast dataset. I also plot some images of WithoutMask.</p>
      </div>
      """,
      unsafe_allow_html=True
    )

    # Training
    st.markdown("#### The Training...")
    
    code2 = """
    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2, shear_range = 0.2,zoom_range=0.2,horizontal_flip=True)
  train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(img_h, img_w),
                                        class_mode="categorical", batch_size=batch_size, subset = "training")

  val_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(img_h, img_w),
                                        class_mode="categorical", batch_size=batch_size, subset="validation")
    """
    st.code(code2, language='python')
    st.write("Above is where I make the train generators using ImageDataGenerator from the Keras API of Tensorflow. I also do the same for the validation generator.")

    code5 = """
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
  mobilenet = MobileNetV2(weights = "imagenet", include_top = False, input_shape=(img_h, img_w, img_channels))

  for layer in mobilenet.layers:
      layer.trainable = False

  model = models.Sequential()
  model.add(mobilenet)
  model.add(layers.Flatten())
  model.add(layers.Dense(2, activation="softmax"))


  model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")
  model.summary()
    """

    st.code(code5, language='python')
    st.write("Above I have a code cell with the intialization and compilation of my custom network for Face Mask Detection.")

    code6 = """
    from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
  checkpoint = ModelCheckpoint("mobilenet_facemask.h5",monitor="val_accuracy", save_best_only=True, verbose=1)
  earlystop = EarlyStopping(monitor="val_acc", patience=5, verbose=1)

  history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator)//32, validation_data=val_generator,
                              validation_steps = len(val_generator)//32, callbacks=[checkpoint,earlystop], epochs=epochs)
    """

    st.code(code6, language='python')
    st.write("Finally, to end off the Training section of this tutorial, here I have the code cell to create the Keras Callbacks and to train the model with the train and validation generators.")

    st.markdown("#### Takeaways from my Code Tutorial")
    st.write("1. Pre-processing: It was applied to all the raw input images to convert them into clean versions, which could be fed to a neural network machine learning model. The input image is resized to 224 x 224 (Line 8) and pass input image to preprocess_input function in , which is meant to adequate your image to the format the model requires (you guarantee that the images you load are compatible with preprocess_input ). Finally converting data and labels to NumPy arrays for further processing in. One hot encoding is performed on labels to convert categorical data to numerical data.")
    st.write("2. Model Creation: The CNN model includes two convolutional layers followed by activation function ReLU(to add non-linearity) and Max Pooling(to reduce the feature map). Dropout is added to Prevent Neural Networks from Over-fitting. Then, fully connected layers are added at the end. Finally, we compiled our model to the loss function, the optimizer, and the metrics. The loss function is used to find error or deviation in the learning process. Keras requires loss function during the model compilation process. Optimization is an important process that optimizes the input weights by comparing the prediction and the loss function and Metrics is used to evaluate the performance of your model.")
    st.write("3. Model Training: Before start training of model we need to split the data into Train-Test data. In our case, there was 90% of training data and 10% of testing data. Models are trained by NumPy arrays using the fit function. The main purpose of this fit function is used to evaluate your model on training.")
    st.write("4. Model Prediction: This is the final step, in which we will evaluate the model’s performance by predicting the test data labels.")

    st.subheader("Experiment Results")
    st.write("1. Analysis of the Custom MobileNetV2 Model")

    # Plotly Graph
    history = {'loss': [4.656362056732178, 0.5887212157249451, 0.5414420962333679, 0.5731006264686584, 0.32412201166152954, 0.00880513060837984, 0.03586995601654053, 0.09114445745944977, 0.2317001074552536, 0.007631943095475435, 0.3561035990715027, 0.09769175201654434, 0.07787705957889557, 0.018362393602728844, 0.04310012236237526],
  'accuracy': [0.6741071343421936, 0.9508928656578064, 0.9553571343421936, 0.9419642686843872, 0.9732142686843872, 0.9955357313156128, 0.9910714030265808, 0.9821428656578064, 0.9821428656578064, 0.9955357313156128, 0.9732142686843872, 0.9910714030265808, 0.9910714030265808, 0.9866071343421936,  0.9955357313156128],
  'val_loss': [3.638742208480835, 0.03143460676074028, 0.34895917773246765, 0.16957125067710876, 2.2351734685344127e-08, 0.5141326785087585, 1.305535806750413e-05, 0.06368743628263474, 0.0365956574678421, 3.725290076417309e-09, 2.9802308176840597e-08, 0.0006714493501931429, 2.3147831598180346e-05, 1.1175869119028903e-08, 4.470345160711986e-08],
  'val_accuracy': [0.75, 0.96875, 0.90625, 0.96875, 1.0, 0.96875, 1.0, 0.96875, 0.96875, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  }

    import chart_studio.plotly as py
    from plotly.graph_objs import Data, Figure

    trace1 = {
    "uid": "9f796594-a39b-11e8-a858-8c8590c988e6", 
    "mode": "lines+markers", 
    "name": "Average Epoch Loss", 
    "type": "scatter", 
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
    "y": [0.6741071343421936, 0.9508928656578064, 0.9553571343421936, 0.9419642686843872, 0.9732142686843872, 0.9955357313156128, 0.9910714030265808, 0.9821428656578064, 0.9821428656578064, 0.9955357313156128, 0.9732142686843872, 0.9910714030265808, 0.9910714030265808, 0.9866071343421936,  0.9955357313156128]
    }

    data = Data([trace1])
    layout = {
    "title": "Training Data Accuracy per Epoch", 
    "xaxis": {"title": "Epoch"}, 
    "yaxis": {"title": "Loss"}
    }

    fig = Figure(data=data, layout=layout)
    st.plotly_chart(fig)

    # Example for image detections
    st.write("2. Implementing face mask detector for images.")
    image3 = Image.open("Screen Shot 2022-06-13 at 10.44.01 AM.png")
    st.image(image3, caption='Another exmaple from my trial runs')

    image4 = Image.open('Screen Shot 2022-06-13 at 10.43.18 AM.png')
    st.image(image4, caption='No Mask example from my trial runs')


    # Conclusion
    st.header("Face Mask Detection Conclusion")
    st.subheader("What\'s Next?")
    st.write('In this article, we have developed CNN based face mask detector, which can contribute to public healthcare, Airports, and Offices to ensure safety. We can also use other training models like VGGNet and MobileNet as base models and using YOLO with darknet to perform face detection.')

    # References
    st.subheader("References")
    st.write("Here are some websites and utilities I used for my project.")
    st.write("1. Images of this website are from Unsplash")
    st.write("2. Lots of API reference from Keras")
    st.write("3. Dataset and Notebook made and from Kaggle")

    st.write("I do not currently have a github page or repository for my project but I believe you may be able to view it in Kaggle. The link is right over here - https://www.kaggle.com/code/srirammuthu/mobilenetv2-face-mask-detection/notebook. But in the Kaggle Notebook page in this notebook you will have access to all of the code.")

  # About Me
  def about():
    st.title("About Me")
    st.write("Hi, I am Sriram Muthu Kumaran (Indian pronunciation). In short, my goal is to create an impressive understanding and Web Application for viewers to understand my project properly. And for some more information on my background is, I'm actually a 7th grader currently in California and I have self-learnt AI using mainly free resources like Coursera and W3Schools and paid like Codecademy.")
    st.write("I am focusing on making AI accessible and improving my knowledge of Computer Vision skills, as this was a project to improve my Computer Vision capabilities. I try to share and explain artificial intelligence in simple terms and share the new research state and applications for everyone using Data Science platforms like Kaggle. I am currently studying some Specializations in artificial intelligence from Coursera - for example, Deep Learning Specialization.")
    st.write("Why did I make a website? Because I love to learn and share what I learn. And this Web App is made with Streamlit and this is my first time using it so in a way I'm learn right now too!")
    st.write("For me, a typical day consists of moving a lot, working on different projects, learning, eating, hanging out with my friends, playing with my dog, going to school, and sleeping at least 8-10 hours. I am particularly interested in AI, Autonomous Vehicles (WAYMO), culinary, and sports in general like basketball which I enjoy very much.")

    st.write("Thank you very much for view my Web App which I put lots of hard work into and I hope you learned as much as I did in this project.")

  # Notebook
  def notebook():
    st.title("My Kaggle Notebook")
    st.write("Here in this section there won\'t be much text since I will only display my code cell fom my ipynb file. If you want to run the code for yourself, in the home page there is a link to my notebook.")

    st.header("The Code...")

    code = """
    import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
import cv2
import random

import os
i = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        i += 1
        if i < 10:
            print(os.path.join(dirname, filename))
    """
    st.markdown("#### Libraries")
    st.code(code, language='python')

    code1 = """
    print(os.listdir("/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset"))
print(os.listdir("/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/Train/"))
    """
    st.markdown("#### EDA (Exploratory Data Anlysis)")
    st.code(code1, language='python')

    code2 = """
    DATA_ROOT = '/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset/'
train_dir = DATA_ROOT + 'Train/'
validation_dir = DATA_ROOT + 'Validation/'
test_dir = DATA_ROOT + 'Test/'
    """
    st.code(code2, language='python')

    code3 = """
    print('Number of images in WithMask section:', len(os.listdir((train_dir + 'WithMask'))))
print('Number of images in WiithoutMask section:', len(os.listdir((train_dir + 'WithoutMask'))))

print('Number of images in WithMask section:', len(os.listdir((validation_dir + 'WithMask'))))
print('Number of images in WiithoutMask section:', len(os.listdir((validation_dir + 'WithoutMask'))))

print('Number of images in WithMask section:', len(os.listdir((test_dir + 'WithMask'))))
print('Number of images in WiithoutMask section:', len(os.listdir((test_dir + 'WithoutMask'))))
    """
    st.code(code3, language='python')

    code4 = """
    def show_withmask(num_imgs):
    plt.figure(figsize=(12, 7))
    for i in range(0, num_imgs):
        rand = random.choice(os.listdir(train_dir + 'WithMask'))
        plt.subplot(1, 5, i+1)
        img = load_img(train_dir + 'WithMask/' + rand)
        plt.xlabel('WithMask')
        plt.imshow(img)
    plt.show()
    
show_withmask(5)
    """
    st.code(code4, language='python')

    code5 = """
    def show_withoutmask(num_imgs):
    plt.figure(figsize=(12, 7))
    for i in range(0, num_imgs):
        rand = random.choice(os.listdir(train_dir + 'WithoutMask'))
        plt.subplot(1, 5, i+1)
        img = load_img(train_dir + 'WithoutMask/' + rand)
        plt.xlabel('WithoutMask')
        plt.imshow(img)
    plt.show()
    
show_withoutmask(5)
    """
    st.code(code5, language='python')

    code6 = """
    epochs = 15
batch_size = 32
img_h = 150
img_w = 150
img_channels = 3
    """
    st.markdown("#### Initialization")
    st.code(code6, language='python')

    code7 = """
    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2, shear_range = 0.2,zoom_range=0.2,horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(img_h, img_w),
                                          class_mode="categorical", batch_size=batch_size, subset = "training")

val_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(img_h, img_w),
                                          class_mode="categorical", batch_size=batch_size, subset="validation")
    """
    st.code(code7, language='python')


    code8 = """
    import warnings
warnings.filterwarnings('ignore')
    """
    st.code(code8, language='python')

    code9 = """
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

mobilenet = MobileNetV2(weights = "imagenet", include_top = False, input_shape=(img_h, img_w, img_channels))

for layer in mobilenet.layers:
    layer.trainable = False

model = models.Sequential()
model.add(mobilenet)
model.add(layers.Flatten())
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics ="accuracy")
model.summary()
    """
    st.markdown("##### DL Model (CNN)")
    st.code(code9, language='python')

    code10 = """
    from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("moblenet_facemask.h5",monitor="val_accuracy",save_best_only=True,verbose=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)

history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator)//32, validation_data=val_generator,
                             validation_steps = len(val_generator)//32, callbacks=[checkpoint,earlystop], epochs=epochs)
    """
    st.code(code10, language='python')

    code11 = """
    model.evaluate_generator(val_generator)

model.save("mobilenetv2_det.h5")
pred = model.predict(val_generator)
pred = np.argmax(pred, axis=1)
pred[:15]
    """
    st.code(code11, language='python')
    
    code12 = """
    from sklearn.metrics import confusion_matrix

plt.figure(figsize = (8,5))
sns.heatmap(confusion_matrix(val_generator.labels, pred.round()), annot = True,fmt="d",cmap = "Blues")
plt.show()
    """
    st.code(code12, language='python')

    code13 = """
    face_model = cv2.CascadeClassifier('../input/casscade-xml/haarcascade_frontalface_default.xml')
    """
    st.markdown("#### Localization and Classifcation")
    st.code(code13, language='python')

    code14 = """
    image_path = '../input/face-mask-12k-images-dataset/Face Mask Dataset/Test/WithMask/1163.png'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4)
out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

for x,y,w,h in faces:
    cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,0,255),1)
plt.figure(figsize=(6,6))
plt.imshow(out_img)
plt.show()
    """
    st.code(code14, language='python')

    code15 = """
    from skimage import io
from scipy.spatial import distance

def detect(image):    
    mask_label = {0:'Has Mask!',1:'No Mask'}
    dist_label = {0:(0,255,0),1:(255,0,0)}
    MIN_DISTANCE = 0

    img= io.imread(image)

    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=8)

    if len(faces)>=1:
        label = [0 for i in range(len(faces))]
        for i in range(len(faces)-1):
            for j in range(i+1, len(faces)):
                dist = distance.euclidean(faces[i][:2],faces[j][:2])
                if dist<MIN_DISTANCE:
                    label[i] = 1
                    label[j] = 1
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
        for i in range(len(faces)):
            (x,y,w,h) = faces[i]
            crop = new_img[y:y+h,x:x+w]
            crop = cv2.resize(crop,(150,150))
            crop = np.reshape(crop,[1,150,150,3])/255.0
            mask_result = model.predict(crop)
            print(mask_label[round(mask_result[0][1])])
            cv2.putText(img, mask_label[round(mask_result[0][1])],(x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_label[label[i]], 2)
            cv2.rectangle(img,(x,y),(x+w,y+h), dist_label[label[i]],1)
        plt.figure(figsize=(10,10))
        plt.imshow(img)

    else:
        print("No Face Detected!")
    """
    st.code(code15, language='python')

    code16 = """
    detect('https://images.unsplash.com/photo-1611637576109-b6f76185ec9b?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mzh8fG1hc2tlZCUyMHBlcnNvbnxlbnwwfHwwfHw%3D&auto=format&fit=crop&w=800&q=60')
    """
    st.markdown("#### Testing on Unsplash Pics")
    st.code(code16, language='python')

    code17 = """
    detect('../input/face-mask-detection5/Photo on 3-12-22 at 1.26 PM 2.jpg')
    """
    st.code(code17, language='python')

    image3 = Image.open("Screen Shot 2022-06-13 at 10.44.01 AM.png")
    st.image(image3, caption='Code Cell Output')

  import numpy as np
  import cv2
  from skimage import io
  from scipy.spatial import distance
  import matplotlib.pyplot as plt
  from tensorflow import keras
  from keras.applications.mobilenet_v2 import preprocess_input
  from keras.preprocessing.image import img_to_array
  from PIL import Image
  import numpy as np
  import cv2
  import os
  model = keras.models.load_model('/content/mobilenetv2_det.h5')
  face_model = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')

  def detect(image):    
      mask_label = {0:'Has Mask!',1:'No Mask'}
      dist_label = {0:(0,255,0),1:(255,0,0)}
      MIN_DISTANCE = 0

      img = io.imread(image)

      img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

      faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=8)

      if len(faces)>=1:
          label = [0 for i in range(len(faces))]
          for i in range(len(faces)-1):
              for j in range(i+1, len(faces)):
                  dist = distance.euclidean(faces[i][:2],faces[j][:2])
                  if dist<MIN_DISTANCE:
                      label[i] = 1
                      label[j] = 1
          new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
          for i in range(len(faces)):
              (x,y,w,h) = faces[i]
              crop = new_img[y:y+h,x:x+w]
              crop = cv2.resize(crop,(150,150))
              crop = np.reshape(crop,[1,150,150,3])/255.0
              mask_result = model.predict(crop)
              print(mask_label[round(mask_result[0][1])])
              cv2.putText(img, mask_label[round(mask_result[0][1])],(x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_label[label[i]], 2)
              cv2.rectangle(img,(x,y),(x+w,y+h), dist_label[label[i]],1)
          st.image(img, caption='Your predicted image')

      else:
          print("No Face Detected!")

  def run_yourself():
    st.title("Run My Deep Learning Model Yourself! - Image Upload")
    st.write("Here in this section of the Streamlit Web App you'll be able to use my face mask detector to test it on some images you manually upload from to do a face mask detection.")
    st.header("Face Mask Detection")
    st.write("Here on this page to the left in the sidebar, you'll see that you can import a file so my MobileNet Model can process it and give a prediction.")

    uploaded_image = st.sidebar.file_uploader("Choose a JPG, JPEG or PNG file", type=["jpg","jpeg","png"])
    confidence_value = st.sidebar.slider('Confidence:', 0.0, 1.0, 0.5, 0.1)

    if uploaded_image:
      st.sidebar.info('Uploaded image:')
      st.sidebar.image(uploaded_image, width=240)
      
    if uploaded_image:
      detect(uploaded_image)

  from streamlit_webrtc import webrtc_streamer, RTCConfiguration
  import av
  
  def run_yourself_webcam():
    st.title("Run My Deep Learning Model Yourself! - Webcam")
    st.write("Here in this section of the Streamlit Web App you'll be able to use my face mask detector to test it on real time webcam frames to do a face mask detection.")
    st.header("Face Mask Detection")
    st.write("Here on this page just click the button below to turn on your webcam and start making real time classifications.")
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    class VideoProcessor:
      def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        mask_label = {0:'Has Mask!',1:'No Mask'}
        dist_label = {0:(0,255,0),1:(255,0,0)}
        MIN_DISTANCE = 0
        
        faces = face_model.detectMultiScale(frm,scaleFactor=1.1, minNeighbors=8)

        if len(faces)>=1:
            label = [0 for i in range(len(faces))]
            for i in range(len(faces)-1):
                for j in range(i+1, len(faces)):
                    dist = distance.euclidean(faces[i][:2],faces[j][:2])
                    if dist<MIN_DISTANCE:
                        label[i] = 1
                        label[j] = 1
            new_img = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR) #colored output image
            for i in range(len(faces)):
                (x,y,w,h) = faces[i]
                crop = new_img[y:y+h,x:x+w]
                crop = cv2.resize(crop,(150,150))
                crop = np.reshape(crop,[1,150,150,3])/255.0
                mask_result = model.predict(crop)
                print(mask_label[round(mask_result[0][1])])
                cv2.putText(frm, mask_label[round(mask_result[0][1])],(x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_label[label[i]], 2)
                cv2.rectangle(frm,(x,y),(x+w,y+h), dist_label[label[i]],1)

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

    webrtc_streamer(key="key", video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration(
              {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
              )
      )

  if choice == 'Home':
    home()

  if choice == 'About':
    about()

  if choice == 'Notebook':
    notebook()

  if choice == 'Run it Yourself - Image':
    run_yourself()

  if choice == 'Run it Yourself - Webcam':
    run_yourself_webcam()

if __name__ == '__main__':
	main()

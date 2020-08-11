# Object-Detection

# Default Object Detection using TF
First install some required library-
pip3 install pillow
pip3 install lxml
pip3 install jupyter
pip3 install matplotlib

Then install the model repo from github.

Run the following command-
protoc object_detection/protos/*.proto --python_out=.
[Make sure you are in object_detection path]
For Linux:
--> cd inside models/research
Run following commands
--> export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim
-->protoc object_detection/protos/*.proto --python_out=.
-->python3 setup.py build
-->sudo python3 setup.py install


Then, go to the object_detection folder and paste the Defaultdetection.py file.
Make sure model path are set full path.
Example-
PATH_TO_LABELS = os.path.join('/home/apubra/Code/GitHub/Object-Detection/Model/models/research/object_detection/data', 'mscoco_label_map.pbtxt')

Then run...


# Custom object detection
You already learned how to run default detection.
You can also detect your own learning object.

Let's try...


First of all you need data for learn.
Manage your data and label it.
We will use popular tool for image label.

Install it in your pc-
Run the following command-
sudo apt-get install pyqt5-dev-tools
sudo pip3 install -r requirements/requirements-linux-python3.txt
make qt5py3
python3 labelImg.py


Then use rectbox, it will make xmin, ymin and xmax, ymax.

Save it in folder.
It will save a xml file.

Ok, now we will convert xml file to csv.

First, make some folder called 
-data 
-images
    -train
    -test
Note:in this train and test folder put your xml and raw image file.

-training
xml to csv.py
generate_tfrecord.py

Then run the converter python file name xml to csv.py

It will transfer your file into csv file in -data folder.


Generate TFRecord-
Then we need generate TFRecord

First create a file name generate_tfrecord.py 
And paste the code from github

Now we have the generate_tfrecord.py file.


You just need some change-

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'jerry':
        return 1
    else:
        None

# Note
I am training jerry image.
So i am using  if row_label == 'jerry':
If you train multiple then just add more if condition.
And return 2, 3 etc...

Andother thing, wahtever image file extension you are using just set it-
image_format = b'png'
Here i am using png


Now run the following command using terminal-

 # Create train data:
  python3 generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=images/train/
  # Create test data:
  python3 generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=images/test/

It will create train.record and test.record file in your data folder.

# Training Custom Object Detector
Ok, Let's try Training Custom Object Detector.

We can train our model from scrach or we can add our object in an existing model.
It's called Transfer Learning. That means whatever object already are trained in that model
plus we are training a new objectin that model.

It's very usefull for build a large model.

Let's apply Transfer learning...

There are mainly two pretrained model in tensorflow:
ssd_mobilenet [Fast] [Good accurate]
faster_r_cnn  [Medium] [More acuurate]

We will use ssd_mobilenet for realtime faster service.
First download the ssd_config file
And then download ssd_model form-
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz


Ok, Lets change the configure file-
num_classes: 1
I am training 1 class. Give whatever class you are training.

batch_size: 24
I am using bydefault

fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt"
Use your model name

input_path: "data/train.record"

label_map_path: "training/object-detection.pbtxt"
Change name, i am using object-detection

eval_input_reader: {
  tf_record_input_reader {
    input_path: "data/test.record"
  }
  label_map_path: "data/object-detection.pbtxt"
  shuffle: false
  num_readers: 1
}

Inside training dir, add object-detection.pbtxt:
item {
  id: 1
  name: 'jerry'
}

num_examples: (number of test images)


Use your training label name.
If you are train more then add more item and just keep it 2,3 etc...

Ok, Then select the following files and folder and paste those in our Tensorflow
model repo in object_detection folder.

data
images
ssd_mobulenet_v1_coco
training
ssd_mobilenet_v1_pets.config


Then move the ssd_mobilenet_v1_pets.config file to training folder in our Tensorflow
model repo in object_detection folder.

It's was our mistake, next time you download ssd_mobilenet_v1_pets.config file in training folder.

Then open terminal in models/research/object_detection location-
Then run the following command-
python3 legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

After run this command you will get an error-
ModuleNotFoundError: No module named 'nets'

Solve the error:
in the /tensorflow1/models/research/slim directory run
python3 setup.py build
sudo python3 setup.py install

models/research/slim HAS ITS OWN setup.py!!!!!!!!!!!!!


Ok, Erveything is ok and it's start training.


From models/object_detection, via terminal, you start TensorBoard with:

tensorboard --logdir='training'

This runs on 127.0.0.1:6006 (visit in your browser)


# Testing Custom Object Detector

Now our loss function is less than 1.
It's good enough for detection.

You can wait for 0.05 below loss.
It will gives you more accuracy.
Now we can stop training...


Ok, Lets Test our train model-
In order to do this, we need to export the inference graph.

Luckily for us, in the models/research/object_detection directory, there is a script that does this for us: export_inference_graph.py

python3 export_inference_graph \
    --input_type image_tensor \
    --pipeline_config_path path/to/ssd_inception_v2.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory



You need to modify the command-
python3 export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-6943 \
    --output_directory jerry_graph

Look at what we modify-
First add the .py in first line.
Second add training/ssd_mobilenet_v1_pets.config our config file path.
Third add the latest trained model. Check your directory latest number are available
in three file.
Example-
training/model.ckpt-6943

And last add your directory name where output go-
output_directory jerry_graph

Ok, run the command from research/object_detection path

Now it wil create folder called jerry_graph and here is our trained model.

Let's run thr model and detect the Object :)




# TF Lite
If you are focusing to run your trained model in low resource hardware such as
Android, Raspberry pi etc you should use the TF Lite model file.
Because it runs more efficient way in low resource platform.

We already learned about How to train our custom object.
But it dosen't gives us TFLite file.
We need to convert TF to TFLite model.


There are three primary steps to training and deploying a TensorFlow Lite model:

1. Train a quantized SSD-MobileNet model using TensorFlow, and export frozen graph for             TensorFlow Lite
2. Build TensorFlow from source on your PC
3. Use TensorFlow Lite Optimizing Converter (TOCO) to create optimzed TensorFlow Lite model



You can also use a standard SSD-MobileNet model (V1 or V2), but it will not run quite as fast as the quantized model. Also, you will not be able to run it on the Google Coral TPU Accelerator.

This tutorial will use the SSD-MobileNet-V2-Quantized-COCO model. Download the model here.
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

Note: TensorFlow Lite does NOT support RCNN models such as Faster-RCNN! It only supports SSD models.

# 1. Train a quantized SSD-MobileNet model using TensorFlow, and export frozen graph for 
So training for TFLite, everything as previous example.
First train your model using quantized SSD model.


Wait for good loss. And we will use that checkpoint to export the frozen TensorFlow Lite graph.

Export frozen inference graph for TensorFlow Lite-
Create a folder name TFLite_model

python3 export_tflite_ssd_graph.py --pipeline_config_path='/home/apubra/Code/GitHub/Object-Detection/Tensorflow/Custom Object Detection/Second Test/SSD Test/models/research/object_detection/training/ssd_mobilenet_v1_pets.config' --trained_checkpoint_prefix='/home/apubra/Code/GitHub/Object-Detection/Tensorflow/Custom Object Detection/Second Test/SSD Test/models/research/object_detection/training/model.ckpt-34103' --output_directory='/home/apubra/Code/GitHub/Object-Detection/Tensorflow/Custom Object Detection/Second Test/SSD Test/models/research/object_detection/TFLite_model' --add_postprocessing_op=true

It will create for us ftlite_graph.pb and tflite_graph.pbtx



We need to .tflite fie.
So, lets build it...

# 2. Build TensorFlow from source on your PC
You can flow the link-
https://www.tensorflow.org/install/source

Run the following command, it will shows bazel error.

Before that you need to install bazel and tensorflow from source--
Step 1: Install required packages
Bazel needs a C++ compiler and unzip / zip in order to work:

sudo apt install g++ unzip zip
If you want to build Java code using Bazel, install a JDK:

# Ubuntu 16.04 (LTS) uses OpenJDK 8 by default:
sudo apt-get install openjdk-8-jdk

# Ubuntu 18.04 (LTS) uses OpenJDK 11 by default:
sudo apt-get install openjdk-11-jdk
Step 2: Run the installer
Next, download the Bazel binary installer named bazel-<version>-installer-linux-x86_64.sh from the Bazel releases page on GitHub.

Download what version your tensorflow need.

Run it as follows:

chmod +x bazel-<version>-installer-linux-x86_64.sh

./bazel-<version>-installer-linux-x86_64.sh --user

The --user flag installs Bazel to the $HOME/bin directory on your system and sets the .bazelrc path to $HOME/.bazelrc. Use the --help command to see additional installation options.



Step 3: Set up your environment
If you ran the Bazel installer with the --user flag as above, the Bazel executable is installed in your $HOME/bin directory. It’s a good idea to add this directory to your default paths, as follows:

export PATH="$PATH:$HOME/bin"
You can also add this command to your ~/.bashrc or ~/.zshrc file to make it permanent.

Ok...

Now you need install tensorflow from source.
Download it from github and run the following command-

You defiently git checkout your tensorflow version.
Please make it sure...


From your branch just type-
./configure

or python3 configure.py

And continue configure TFLite extra features.

And last you need run the command for cpu only-
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

# 3. Use TensorFlow Lite Optimizing Converter (TOCO) to create optimzed TensorFlow Lite model

After that run the following command from your tnsorflow directory.
My location is now-
/home/apubra/Code/GitHub/Object-Detection/TensorflowLite/ConvertingTFtoTFLite/tensorflow

Open the terminal from here and run the command-

If you are using a floating, non-quantized SSD model then run the command-

bazel run --config=opt tensorflow/lite/toco:toco -- --input_file='/home/apubra/Code/GitHub/Object-Detection/Tensorflow/Custom Object Detection/Second Test/SSD Test/models/research/object_detection/TFLite_model/tflite_graph.pb' --output_file='/home/apubra/Code/GitHub/Object-Detection/Tensorflow/Custom Object Detection/Second Test/SSD Test/models/research/object_detection/TFLite_model/detect.tflite' --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --inference_type=FLOAT --allow_custom_ops 

Or, If you are using a quantized SSD model then run the command-

bazel run --config=opt tensorflow/lite/toco:toco -- --input_file=%OUTPUT_DIR%/tflite_graph.pb --output_file=%OUTPUT_DIR%/detect.tflite --input_shapes=1,300,300,3 --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_values=128 --change_concat_input_ranges=false --allow_custom_ops 

Note: Just change your path


After the command finishes running, you should see a file called detect.tflite in the \object_detection\TFLite_model directory. This is the model that can be used with TensorFlow Lite!


# Create new label map

For some reason, TensorFlow Lite uses a different label map format than classic TensorFlow.

However, the label map provided with the example TensorFlow Lite object detection model looks like this:
person 
bicycle 
car 
motorcycle 
And so on...

Thus, we need to create a new label map that matches the TensorFlow Lite style. Open a text editor and list each class in order of their class number. Then, save the file as “labelmap.txt” in the TFLite_model folder.

This label sequence will be same as what sequence you learned already.
retur 1,2,3 etc...

# Note
I am reminder again, first check your tensorflow fersion.
Then download the required bazel version, which is for your tensorflow version and install it.
Again make sure you are installing bazel from your tensorflow version git branch.
Now this time i am using tensorflow 1.14
so my branch will be r1.14

Then add the export PATH="$PATH:$HOME/bin" in your .bashrc file.
It makes your bazel accessable from globaly.

Then configure your tensorflow properly.

Than run that command from tensorflow directory.
That's it. It will make a detect.tflite file for us :)


# Run your TFLite from your pc or raspberry pi
python3 TFLite_detection_webcam.py --modeldir=TFLite_model

# Run your TFLite from Android


We followed the git repo for this document-
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi#step-1d-export-frozen-inference-graph-for-tensorflow-lite
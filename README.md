This repository contains the code for the following papers:

* \[1\] Francesco Visin, Kyle Kastner, Kyunghyun Cho, Matteo Matteucci, Aaron
        Courville, Yoshua Bengio - [ReNet: A Recurrent Neural Network Based
        Alternative to Convolutional Networks](
        https://arxiv.org/pdf/1505.00393.pdf) ([BibTeX](
        https://scholar.google.it/scholar.bib?q=info:ZUE20uoSIzcJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAV0XzMYPJjoJeUCn69TuCoJkSaHSAOnCj&scisf=4&hl=en))

* \[2\] Francesco Visin, Kyle Kastner, Aaron Courville, Yoshua Bengio, Matteo
        Matteucci, Kyunghyun Cho - [ReSeg: A Recurrent Neural Network for
        Object Segmentation](http://arxiv.org/pdf/1511.07053) ([BibTeX](
        https://scholar.google.it/scholar.bib?q=info:LHt-RjBaDBEJ:scholar.google.com/&output=citation&scisig=AAGBfm0AAAAAV0XywMgiCDP11cRAVEyL5otAP2YnQMdo&scisf=4&hl=en)) 

Setup
-----

#### Install Theano

Download Theano and make sure it's working properly.  All the
information you need can be found by following this link:
http://deeplearning.net/software/theano/

#### Download the CamVid dataset

Download the CamVid dataset from 
http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

The program expects to find the dataset data in `./datasets/camvid/`. You can
change this path modifying `camvid.py` if you want.

#### Download the VGG-16 weights
Download the VGG weights for Lasagne from:
https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

Once downloaded, rename them as `w_vgg16.pkl` and put them in the root
directory of this code.

Reproducing the Results
-----------------------

To reproduce the results of the ReSeg paper run `python evaluate_camvid.py`.
Make sure to set the appropriate THEANO_FLAGS to run the model on your machine
(most probably `export THEANO_FLAGS=device=gpu,floatX=float32`)

The program will output some metrics on the current minibatch iteration during
training:

    Epoch 0/5000 Up 367 Cost 270034.031250, DD 0.000046, UD 0.848205 
    (None, 360, 480, 3)

More in detail, it will show the current epoch, the incremental update counter
(i.e. number of minibatches seen), the cost of the current iteration, the time
(in seconds) required to load the data `DD` and to train and update the
network's parameters `DD`. Finally, it will print the size of the currently
processed minibatch. `None` will be displayed on variable-sized dimensions.

At the end of each epoch, it will validate the performances on the training,
validation and test set and save some sample images for each set in a
*segmentations* directory inside the root directory of the script.

At the end of the training `get_info_model.py` can be used to show some
information on the trained model. Run `python get_info_model.py -h` for a 
list of the arguments and their explanation.

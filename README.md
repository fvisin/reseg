If you use this code, please cite one of the following papers:

* \[1\] Francesco Visin, Kyle Kastner, Kyunghyun Cho, Matteo Matteucci, Aaron
        Courville, Yoshua Bengio - [ReNet: A Recurrent Neural Network Based
        Alternative to Convolutional Networks](
        https://arxiv.org/pdf/1505.00393.pdf) ([BibTeX](
        https://gist.github.com/fvisin/e450c4f55a527c5db802e69574b79a95#file-renet-bib))

* \[2\] Francesco Visin, Marco Ciccone, Adriana Romero, Kyle Kastner, Kyunghyun 
        Cho, Yoshua Bengio, Matteo Matteucci, Aaron Courville - [ReSeg: A Recurrent 
        Neural Network-based Model for Semantic Segmentation](
        http://arxiv.org/pdf/1511.07053) ([BibTeX](
        https://gist.github.com/fvisin/61b1dd3777ea91a0e3ad963366a61fb1#file-reseg-bib))


Setup
-----

#### Install Theano

Download Theano and make sure it's working properly.  All the
information you need can be found by following this link:
http://deeplearning.net/software/theano/


#### Install other dependencies

This software relies on some amazing third-party software libraries. 
You can install them with *pip*:
`pip install <--user> lasagne matplotlib Pillow progressbar2 pydot-ng retrying
scikit-image scikit-learn tabulate`
*(Use the `--user` option if you don't want to install them globally or you
don't have sudo privileges on your machine.)*


#### Download the CamVid dataset

Download the CamVid dataset from 
http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

Resize the images to 480X360 resolution. The program expects to find the 
dataset data in `./datasets/camvid/`. You can change this path modifying 
`camvid.py` if you want.


#### Download the VGG-16 weights
Download the VGG weights for Lasagne from:
https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl

Once downloaded, rename them as `w_vgg16.pkl` and put them in the root
directory of this code.


Reproducing the Results
-----------------------

To reproduce the results of the ReSeg paper run `python evaluate_camvid.py` (or
`python evaluate_camvid_with_cb.py` to reproduce the experiment with class
balancing).  Make sure to set the appropriate THEANO_FLAGS to run the model on
your machine (most probably `export THEANO_FLAGS=device=gpu,floatX=float32`)

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

****
Note: In case you want to modify this code to reproduce the results of 
"Combining the best of convolutional layers and recurrent layers: A hybrid
network for semantic segmentation" please let us know!


Acknowledgments
---------------

Many people contributed in different ways to this project. We are extremely
thankful to the [Theano](http://deeplearning.net/software/theano/) developers
and to many people at [MILA](http://mila.umontreal.ca/) for their support and
for the many insightful discussions. We also thank the developer of
[Lasagne](http://lasagne.readthedocs.io/), a powerful yet light framework on top of
Theano. I wish I discovered it at the beginning of this project! :)

Finally, our gratitude goes to the developers of all the great libraries we
used in this project, to all the people who got involved with the project at
any level and to our generous sponsors.

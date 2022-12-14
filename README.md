# gan-segmentation-person-gun
GAN segmentation gun person
This repository contains code to accompany the O'Reilly tutorial on generative adversarial networks written by Jon Bruner and Adit Deshpande. See the original tutorial to run this code in a pre-built environment on O'Reilly's servers with cell-by-cell guidance, or run these files on your own machine.

There are three versions of our simple GAN model in this repository:

gan-notebook.ipynb is identical to the interactive tutorial, available here so that you can run it on your own machine.
gan-script.py is a straightforward Python script containing code drawn directly from the tutorial, to be run from the command line. Note that it doesn't print anything when it's executed, but it does send regular updates to TensorBoard so that you can track its progress.
gan-script-fast.py is a modest refactoring of gan-script.py that runs slightly faster because more of its computations are contained in the TensorFlow graph.
Requirements and installation
In order to run gan-script.py or gan-script-fast.py, you'll need TensorFlow version 1.0 or later and NumPy. In order to run gan-notebook.ipynb, you'll additionally need Jupyter and matplotlib.

If you've already got TensorFlow on your machine, then you've got NumPy and should be able to run the raw Python scripts.

Installing Anaconda Python and TensorFlow
The easiest way to install TensorFlow as well as NumPy, Jupyter, and matplotlib is to start with the Anaconda Python distribution.

Follow the installation instructions for Anaconda Python. We recommend using Python 3.6.

Follow the platform-specific TensorFlow installation instructions. Be sure to follow the "Installing with Anaconda" process, and create a Conda environment named tensorflow.

If you aren't still inside your Conda TensorFlow environment, enter it by opening your terminal and typing

source activate tensorflow
Download and unzip this entire repository from GitHub, either interactively, or by entering

git clone https://github.com/jonbruner/generative-adversarial-networks.git
Use cd to navigate into the top directory of the repo on your machine

Launch Jupyter by entering

jupyter notebook
and, using your browser, navigate to the URL shown in the terminal output (usually http://localhost:8888/)
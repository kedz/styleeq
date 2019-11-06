# styleeq
Code for Low-Level Linguistic Controls for Style Transfer and Content Preservation

# INSTALL LIBRARY

First install plum:

$ python setup.py install

Then setup the eval scripts:

$ cd eval_scripts; ./install.sh; cd ..

# DOWNLOAD DATA

./download_data.sh

Data will apear in a directory called literary_style_data.

# Train 

To train the StyleEQ model run:

>>> plumr configs/styleeq.jsonnet --proj models/styleeq --run train --gpu GPUNUM

where GPUNUM is the number of the gpu you want to run on. -1 will run on cpu
but this is not practical. 

To evaluate the model with automatic quality metrics on the test, after 
training, run: 

>>> plumr configs/styleeq.jsonnet --proj models/styleeq --run eval-test --gpu GPUNUM

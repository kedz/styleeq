# styleeq
Code for Low-Level Linguistic Controls for Style Transfer and Content Preservation

# INSTALL LIBRARY

First install plum:

`$ python setup.py install`

Then download the spacy model data:

`$ python -m spacy download en_core_web_sm`

Then setup the eval scripts:

`$ cd eval_scripts; ./install.sh; cd ..`

# DOWNLOAD DATA

`./download_data.sh`

Data will apear in a directory called literary_style_data.

# DOWNLOAD MODELS

Instead of training your own models, you can use our models we used for the
paper by downloading them. Simply run:

`./download_models.sh`

Models will appear in a directory called style_models.

# Train 

To train the StyleEQ model run:

`$ plumr configs/styleeq.jsonnet --proj style_models/styleeq --run train --gpu GPUNUM`

where GPUNUM is the number of the gpu you want to run on. -1 will run on cpu
but this is not practical. 

To evaluate the model with automatic quality metrics on the test, after 
training, run: 

`$ plumr configs/styleeq.jsonnet --proj style_models/styleeq --run eval-test --gpu GPUNUM`

# Generation Example

To see how to generate text/perform style transfer see the example generation 
script, generation_example.py.

This gives an example of generating from data already in the format of 
the jsonl data that you can download above, and how to convert a raw string
to that format. To convert a raw string to the correct format requires 
the Stanford Core NLP library. To download run:

`$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip` 

`$ unzip stanford-corenlp-full-2018-10-05.zip`

To run the generation script, run:

`$ python generation_example.py style_models/styleeq/ literary_style_data stanford-corenlp-full-2018-10-05`


 




# styleeq
Code for Low-Level Linguistic Controls for Style Transfer and Content Preservation

To train the StyleEQ model run:

>>> plumr configs/styleeq.jsonnet --proj models/styleeq --run train --gpu GPUNUM

where GPUNUM is the number of the gpu you want to run on. -1 will run on cpu
but this is not practical. 

To evaluate the model with automatic quality metrics on the test, after 
training, run: 

>>> plumr configs/styleeq.jsonnet --proj models/styleeq --run eval-test --gpu GPUNUM

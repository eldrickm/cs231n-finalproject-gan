![kith](https://github.com/eldrickm/cs231n-finalproject-gan/blob/master/hUNAo-1.png)

Setup
- wget https://pjreddie.com/media/files/yolov3.weights

32 x 32 Downsampled MS COCO (eventually use full size 2017 data)
- cd inpainting
- wget http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/inpainting.tar.bz2 
- tar xvjf inpainting.tar.bz2
should have two folders "train2014" and "val2014"

- cd inpainting/annotations
- wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

Fist Time Install cocoapi + Create Conda Environment (if not built already)
- cd WGAN
- git clone https://github.com/cocodataset/cocoapi.git
- conda create -n cs231n python=3.6 anaconda
- conda activate cs231n
- cd cocoapi/PythonAPI
- make
- sudo make install
- python setup.py install

Afterwards to run notebook (always run)
- conda activate cs231n

# ColorFool

This is the official repository of [ColorFool: Semantic Adversarial Colorization](https://arxiv.org/pdf/1911.10891.pdf), a work published in the Proc. of the Conference on Computer Vision and Pattern Recognition, (<b>CVPR</b>), Seattle, Washington, USA, 14-19 June, 2020.<br>


<b>Example of results</b>

| Original Image | Attack AlexNet | Attack ResNet18 | Attack ResNet50 |
|---|---|---|---|
| ![Original Image](Dataset/ILSVRC2012_val_00003533.JPEG) | ![Attack AlexNet](Adversarial/Samples/ILSVRC2012_val_00003533_alexnet.png) |![Attack ResNet18](Adversarial/Samples/ILSVRC2012_val_00003533_resnet18.png) | ![Attack ResNet50](Adversarial/Samples/ILSVRC2012_val_00003533_resnet50.png) |


## Setup
1. Download source code from GitHub
   ```
    git clone https://github.com/smartcameras/ColorFool.git 
   ```
2. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name ColorFool python=2.7.15
   ```
3. Activate conda environment
   ```
    source activate ColorFool
   ```
4. Install requirements
   ```
    pip install -r requirements.txt
   ```



## Description
The code works in two steps: 
1. Identify image regions using semantic segmentation model
2. Generate adversarial images via perturbing color of semantic regions in the natural color range    


### Semantic Segmentation 

We identify four semantic regions whose color is important for a human observer as the appearance of these sensitive regions is typically within a specific range including person, sky, vegetation (grass), and water (sea, river, waterfall, swimming pool, and lake). We perform semantic segmentation, using [Pyramid Pooling R50-Dilated architecture of Cascade Segmentation Module](https://github.com/CSAILVision/semantic-segmentation-pytorch) trained on MIT ADE20K dataset, as follows: 

1. Go to Segmentation directory
   ```
   cd Segmentation
   ```
2. Download segmentation model (i.e. both encoder and decoder)
   ```
   wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth -P baseline-resnet50dilated-ppm_deepsup/
   wget http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth -P baseline-resnet50dilated-ppm_deepsup/ 
   ```   
3. Find the sensitive semantic regions within an image
   ```
   bash script.sh
   ```

The semantic regions of four categories will be saved in the Segmentation/SegmentationResults/$Dataset/ directory as a smooth mask the same size of the image with the same name as their corresponding original images

### Generate ColorFool adversarial images

After saving semantic masks of sensitive regions, we converts the intensities of the clean image from the
RGB to the perceptually uniform Lab color space to modify randomly the colors of each region via changing a and b channels in the set of natural-color ranges. We allow trials up to 1000, until a perturbation misleads the classifier. ColorFool adversarial images are then generated


1. Go to Adversarial directory
   ```
   cd ../Adversarial
   ```
2. In the script.sh set 
(i) the name of target models for attack, and (ii) the name of the dataset.
The current implementation supports three classifiers Resnet18, Resnet50 and Alexnet trained in ImageNet. Other classifiers can be employed by modifying initialise function in misc_functions.py.
3. Set the path of the Dataset and masks of all four semantic regions in the initialise function of misc_functions.py and ColorFool.py
4. Generate ColorFool adversarial images 
   ```
   bash script.sh
   ```



## Authors
* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Ricardo Sanchez-Matilla](mailto:ricardo.sanchezmatilla@qmul.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)


## References
If you use our code, please cite the following paper:

      @InProceedings{shamsabadi2020colorfool,
        title = {ColorFool: Semantic Adversarial Colorization},
        author = {Shamsabadi, Ali Shahin and Sanchez-Matilla, Ricardo and Cavallaro, Andrea},
        booktitle = {Proceedings of the Conference on Computer Vision and Pattern Recognition, (CVPR)},
        year = {2020},
        address = {Seattle, Washington, USA},
        month = June
      }
## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).

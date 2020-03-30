# ColorFool

This is the official repository of [ColorFool: Semantic Adversarial Colorization](https://arxiv.org/pdf/1911.10891.pdf), a work published in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, Washington, USA, 14-19 June, 2020.<br>


<b>Example of results</b>

| Original Image | Attack AlexNet | Attack ResNet18 | Attack ResNet50 |
|---|---|---|---|
| ![Original Image](Dataset/ILSVRC2012_val_00003533.JPEG) | ![Attack AlexNet](Sample_results/ILSVRC2012_val_00003533_alexnet.png) |![Attack ResNet18](Sample_results/ILSVRC2012_val_00003533_resnet18.png) | ![Attack ResNet50](Sample_results/ILSVRC2012_val_00003533_resnet50.png) |


## Setup
1. Download source code from GitHub
   ```
    git clone https://github.com/smartcameras/ColorFool.git 
   ```
2. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
    conda create --name ColorFool python=3.5.6
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

1. Go to Segmentation directory
   ```
   cd Segmentation
   ```
2. Download segmentation model (both encoder and decoder) ([source](https://github.com/CSAILVision/semantic-segmentation-pytorch))
   ```
   wget http://www.eecs.qmul.ac.uk/~rsm31/data/CVPR2020/decoder_epoch_20.pth -P models/
   wget http://www.eecs.qmul.ac.uk/~rsm31/data/CVPR2020/encoder_epoch_20.pth -P models/ 
   ```   
3. Run the segmentation for all images within Dataset directory (requires GPU)
   ```
   bash script.sh
   ```

The semantic regions of four categories will be saved in the Segmentation/SegmentationResults/$Dataset/ directory as a smooth mask the same size of the image with the same name as their corresponding original images

### Generate ColorFool Adversarial Images

1. Go to Adversarial directory
   ```
   cd ../Adversarial
   ```
2. In the script.sh set 
(i) the name of target models for attack, and (ii) the name of the dataset.
The current implementation supports three classifiers (Resnet18, Resnet50 and Alexnet) trained with ImageNet.
3. Run ColorFool for all images within the Dataset directory (works in both GPU and CPU)
   ```
   bash script.sh
   ```

### Outputs
* Adversarial Images saved with the same name as the clean images in Adversarial/Results/ColorFoolImgs directory;
* Metadata with the following structure: filename, number of trials, predicted class of the clean image with its probablity and predicted class of the adversarial image with its probablity in Adversarial/Results/Logs directory.


## Authors
* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk)
* [Ricardo Sanchez-Matilla](mailto:ricardo.sanchezmatilla@qmul.ac.uk)
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk)


## References
If you use our code, please cite the following paper:

      @InProceedings{shamsabadi2020colorfool,
        title = {ColorFool: Semantic Adversarial Colorization},
        author = {Shamsabadi, Ali Shahin and Sanchez-Matilla, Ricardo and Cavallaro, Andrea},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2020},
        address = {Seattle, Washington, USA},
        month = June
      }

## License
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

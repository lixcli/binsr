# introduction

a simple binary neural network for SISR(base on EDSR official [code](https://github.com/sanghyun-son/EDSR-PyTorch))

if you want to try different binary function, you need to :
1. add new script to model/binarize and modify model/__init__.py to choose your binary function.
2. edit [run.sh](./run.sh)(view this script for more detail), choose one function you want to run.
3. run `sh run.sh`


# performance
an example binary function:
`bianrize/WApproxASTE.py`:
# weight binarizing

forward:

<img src="https://latex.codecogs.com/svg.image?\operatorname{sign}(x)=&space;\begin{cases}-1&space;&&space;\text&space;{&space;if&space;}&space;x<0&space;\\&space;&plus;1&space;&&space;\text&space;{&space;otherwise&space;}\end{cases}" title="\operatorname{sign}(x)= \begin{cases}-1 & \text { if } x<0 \\ +1 & \text { otherwise }\end{cases}" />



backward,use approxsign ([bireal-net's method](https://openaccess.thecvf.com/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf))


<img src="https://latex.codecogs.com/svg.image?F(\mathrm{X})=\left\{\begin{array}{cc}-1&space;&&space;\text&space;{&space;if&space;}&space;\mathrm{x}<-1&space;\\2&space;x&plus;\mathrm{x}^{2}&space;&&space;\text&space;{&space;if&space;}-1&space;\leqslant&space;\mathrm{x}<0&space;\\2&space;x-\mathrm{x}^{2}&space;&&space;\text&space;{&space;if&space;}&space;0&space;\leqslant&space;\mathrm{x}<1&space;\\1&space;&&space;\text&space;{&space;otherwise&space;}\end{array},&space;\frac{\partial&space;F\left(a_{r}\right)}{\partial&space;a_{r}}=\left\{\begin{array}{cc}2&plus;2&space;x&space;&&space;\text&space;{&space;if&space;}-1&space;\leqslant&space;\mathrm{x}<0&space;\\2-2&space;\mathrm{x}&space;&&space;\text&space;{&space;if&space;}&space;0&space;\leqslant&space;\mathrm{x}<1&space;\\0&space;&&space;\text&space;{&space;otherwise&space;}\end{array}\right.\right." title="F(\mathrm{X})=\left\{\begin{array}{cc}-1 & \text { if } \mathrm{x}<-1 \\2 x+\mathrm{x}^{2} & \text { if }-1 \leqslant \mathrm{x}<0 \\2 x-\mathrm{x}^{2} & \text { if } 0 \leqslant \mathrm{x}<1 \\1 & \text { otherwise }\end{array}, \frac{\partial F\left(a_{r}\right)}{\partial a_{r}}=\left\{\begin{array}{cc}2+2 x & \text { if }-1 \leqslant \mathrm{x}<0 \\2-2 \mathrm{x} & \text { if } 0 \leqslant \mathrm{x}<1 \\0 & \text { otherwise }\end{array}\right.\right." />

# activation binarizing
forward:

<img src="https://latex.codecogs.com/svg.image?\operatorname{sign}(x)=&space;\begin{cases}-1&space;&&space;\text&space;{&space;if&space;}&space;x<0&space;\\&space;&plus;1&space;&&space;\text&space;{&space;otherwise&space;}\end{cases}" title="\operatorname{sign}(x)= \begin{cases}-1 & \text { if } x<0 \\ +1 & \text { otherwise }\end{cases}" />

backward,STE

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\operatorname{clip}(-1,&space;x,&space;1)}{\partial&space;X}=\left\{\begin{array}{c}1,-1<x<1&space;\\0,&space;\text&space;{&space;otherwise&space;}\end{array}\right." title="\frac{\partial \operatorname{clip}(-1, x, 1)}{\partial X}=\left\{\begin{array}{c}1,-1<x<1 \\0, \text { otherwise }\end{array}\right." />



# network modify

**add an Tanh() function before binact**. It helps to transfer a nonlinear gradient after STE.


## srresnet

|Method|Bit|Set5|Set14|B100|Urban100|
|:----|:----|:----|:----|:----|:----|
| |W/A|PSNR/SSIM|PSNR/SSIM|PSNR/SSIM|PSNR/SSIM|
|srresnetx2 |32/32|37.889/0.958|33.4/0.915|32.077/0.896|31.602/0.922|
|Oursx2|1/1|36.345/0.934| 32.221/0.876|31.364/0.877|29.407/0.883|
|FPx4|32/32|32.066/0.890|28.497/0.778|27.516/0.731|25.858/0.778|
|Oursx4|1/1|31.232/0.848|28.047/0.729|27.215/0.699|25.075/ 0.727|

compare to [Efficient Super Resolution Using Binarized Neural Network](http://openaccess.thecvf.com/content_CVPRW_2019/papers/CEFRL/Ma_Efficient_Super_Resolution_Using_Binarized_Neural_Network_CVPRW_2019_paper.pdf)(Ma et.al.)

|Method|Bit|Set5|Set14|Urban100|
|:----|:----|:----|:----|:----|
| |W/A|PSNR|PSNR|PSNR|PSNR|
|Ma et.al. x2|1/1|35.66|31.56|28.76|
|Oursx2|1/1|36.345| 32.221|29.407|
|Ma et.al. x4|1/1|30.34|27.16|24.48|
|Oursx4|1/1|31.232|28.047|25.075|

## edsr

|Method|Bit|Set5|Set14|B100|Urban100|
|:----|:----|:----|:----|:----|:----|
| |W/A|PSNR/SSIM|PSNR/SSIM|PSNR/SSIM|PSNR/SSIM|
|edsrx2 |32/32|37.931/0.958|33.459/0.915|32.102/0.896|31.709/0.923|
|Oursx2|1/1|37.262/0.940|32.826/0.886|31.629/0.882|30.165/0.896|
|edsrx4|32/32|32.095/0.890|28.576/0.780|27.562/0.732|26.035/0.784|
|Oursx4|1/1|30.887/0.839|27.791/0.721|27.041/0.693|24.723/0.712|





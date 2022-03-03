
a simple binary neural network for SISR(base on EDSR official [code](https://github.com/sanghyun-son/EDSR-PyTorch))

if you want to try different binary function, you need to :
1. add new script to model/binarize and model/__init__.py to choose your binary function.
2. edit [run.sh](./run.sh)(view this script for more detail), choose one function you want to run.
3. run `sh run.sh`


# performance
a example binary function:
`bianrize/WApproxASTE.py`:
# weight binarizing

forward:

$$
\operatorname{sign}(x)= \begin{cases}-1 & \text { if } x<0 \\ +1 & \text { otherwise }\end{cases}
$$


backward,use approxsign[bireal-net](https://openaccess.thecvf.com/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf)
$$
F(\mathrm{X})=\left\{\begin{array}{cc}
-1 & \text { if } \mathrm{x}<-1 \\
2 x+\mathrm{x}^{2} & \text { if }-1 \leqslant \mathrm{x}<0 \\
2 x-\mathrm{x}^{2} & \text { if } 0 \leqslant \mathrm{x}<1 \\
1 & \text { otherwise }
\end{array}, \frac{\partial F\left(a_{r}\right)}{\partial a_{r}}=\left\{\begin{array}{cc}
2+2 x & \text { if }-1 \leqslant \mathrm{x}<0 \\
2-2 \mathrm{x} & \text { if } 0 \leqslant \mathrm{x}<1 \\
0 & \text { otherwise }
\end{array}\right.\right.
$$

# activation binarizing
forward:

$$
\operatorname{sign}(x)= \begin{cases}-1 & \text { if } x<0 \\ +1 & \text { otherwise }\end{cases}
$$

backward,STE

$$
\frac{\partial \operatorname{clip}(-1, x, 1)}{\partial X}=\left\{\begin{array}{c}
1,-1<x<1 \\
0, \text { otherwise }
\end{array}\right.
$$

# network modify

add an Tanh() function before binact. It helps to transfer a nonlinear gradient after STE.


## srresnet

|Method|Bit|Set5|Set14|B100|Urban100|
|:----|:----|:----|:----|:----|:----|
| |W/A|PSNR/SSIM|PSNR/SSIM|PSNR/SSIM|PSNR/SSIM|
|srresnetx2 |32/32|37.889/0.958|33.4/0.915|32.077/0.896|31.602/0.922|
|Oursx2|1/1|36.345/0.934| 32.221/0.876|31.364/0.877|29.407/0.883|
|FPx4|32/32|32.066/0.890|28.497/0.778|27.516/0.731|25.858/0.778|
|srresnetx4|1/1|31.232/0.848|28.047/0.729|27.215/0.699|25.075/ 0.727|

## edsr

|Method|Bit|Set5|Set14|B100|Urban100|
|:----|:----|:----|:----|:----|:----|
| |W/A|PSNR/SSIM|PSNR/SSIM|PSNR/SSIM|PSNR/SSIM|
|edsrx2 |32/32|37.931/0.958|33.459/0.915|32.102/0.896|31.709/0.923|
|Oursx2|1/1|37.262/0.940|32.826/0.886|31.629/0.882|30.165/0.896|
|edsrx4|32/32|32.095/0.890|28.576/0.780|27.562/0.732|26.035/0.784|
|Oursx4|1/1|30.887/0.839|27.791/0.721|27.041/0.693|24.723/0.712|





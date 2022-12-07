Program to interactively process magnetic vertical gradient data.
The original raw data was no longer available instead a synthetic dataset was made from an image.
De data is split in 9 seperate files (surveys) which are shown here:
![Raw data](https://github.com/Ohnoj/Geophysics/blob/main/MagneticGradiometry/RawData.png?raw=true)
The program can then be used to apply destaggering (by pressing 'd'), zero mean traverse ('z'), edge matching ('e') on individual sets. After which interpolation ('i') and low-pass filter ('l') can be applied to the full image. The final result will look something like this:
![Results](https://github.com/Ohnoj/Geophysics/blob/main/MagneticGradiometry/Result.png?raw=true)

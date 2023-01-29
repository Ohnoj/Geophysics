# Marchenko 1D
The following scripts are available:
   1. Functions1D.py: contains the essential functions:
      - Layercac computes the response of a 1D acoustic medium
      - tdeps creates windowing operator
      - ricker creates a ricker wavelet
      - flatwave creates a flat wavelet
      - conv either convolves or correlates two traces A and B
   2. Marchenko_example.py: applies the Marchenko equations in 1D, provides redatumed (by deconvolution of the Green's functions) responses and overburden removal (by deconvolution of the focusing functions)
   3. Marchenko_uv_example.py: same as Marchenko_example.py but with the extrapolated functions (e.g. van der Neut 2016)

# Velocity model:
![Model](https://github.com/Ohnoj/Geophysics/blob/main/Marchenko1D/Model.png?raw=true)
# Extrapolated focusing functions v+ and v-:
![FocusingFunctions](https://github.com/Ohnoj/Geophysics/blob/main/Marchenko1D/FocusingFunctions.png?raw=true)
# Underburden removal via Marchenko and direct modelling:
![Removal](https://github.com/Ohnoj/Geophysics/blob/main/Marchenko1D/UnderburdenRemoval.png?raw=true)

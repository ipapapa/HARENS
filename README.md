# Hardware Accelarated Redundancy Elimination in Network Systems

We have developed a Hardware Accelarated Redundancy Elimination in Network Systems (HARENS). It is a software that significantly improve the performance of redundancy elimination algorithms, through the use of Graphics Processing Unit (GPU) acceleration,  hierarchical multi-threaded pipeline, single machine map-reduce, and other memory efficiency techniques to obtain an optimum performance of the redundancy elimination algorithm.

This project has been developed by [Kelu Diao](mailto:keludiao@gmail.com) and [Dr. Ioannis Papapanagiotou](mailto:ipapapa@ncsu.edu).

##Requirements
- Windows 7/8 (Windows 10 possibly doesn't support CUDA architecture)
- 64-bit operating system
- NVidia GPU
- CUDA 6.5 driver or later
- (For Windows users)
  - Visual Studio 2013 or later
  - C++ 11

##Usage Guide
###Installations
- (For Windows users)
  - Open HARENS.sln with Visual Studio and compile
  - The default setting is release-x64, you would find "HARENS.exe", the executable, in path .\x64\Release\
- (For Linux users)
  - If apt-get is supported, type the folling command
    - ```cd \path\to\Project\HARENS```
    - ```make clean``` (only when you want to re-install)
    - ```make install```
    - ```make```
  - If apt-get is not supported
    - Edit makefile (I'm sure you can handle this)
    - ```make install```
    - ```make```
  - You would find "run", the executable, in current path

###Usage
Refer to the guide by typing:
- (For Windows users)
  - type ```HARENS.exe -h```
- (For Linux users)
  - type ```run -h```

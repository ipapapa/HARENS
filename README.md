# Hardware Accelarated Redundancy Elimination in Network Systems

We have developed a Hardware Accelarated Redundancy Elimination emulation framework for Network Systems. Our goal is to optimize the performance of Data Redundancy Elimination (DRE) systems that such they can be applied in higher bandwidth network links. We named our systems HARENS. HARENS significantly improves the performance of redundancy elimination algorithms, through 
- the use of Graphics Processing Unit (GPU) acceleration,  
- hierarchical multi-threaded pipeline, 
- single machine map-reduce, 
- and other memory efficiency techniques.

Our results indicate that HARENS can increase the throughput of other CUDA Redundancy Elimination Systems by x7 up to speeds of 2.5Gbps.
![throughput](https://cloud.githubusercontent.com/assets/4562887/11403658/e47858c8-9352-11e5-80c4-af876147ea8b.png)

This project has been developed by [Kelu Diao](mailto:keludiao@gmail.com) and [Dr. Ioannis Papapanagiotou](mailto:ipapapa@ncsu.edu).

##Requirements
- Windows 7/8 64-bit (Windows 10 possibly doesn't support CUDA architecture) or Linux 64-bit
- NVidia GPU, which supports cuda compute capability 3.5 or later
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

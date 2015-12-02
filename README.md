# Hardware Accelarated Redundancy Elimination in Network Systems

[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)

With the tremendous growth in the amount of information stored on remote locations and cloud systems, many service providers are seeking ways to reduce the amount of redundant information sent across networks by using data de-duplication techniques. Data de-duplication can reduce network traffic without loss of information, and therefore increase available network bandwidth. However, due to heavy computations, de-duplication itself can become a bottleneck in high capacity links. Here, we propose a method named Hardware Accelerated Redundancy Elimination in Network Systems (HARENS). HARENS can significantly improve the performance of the redundancy elimination in network system by leveraging:
- the use of General Purpose Graphics Processing Unit (GPGPU) acceleration,  
- hierarchical multi-threaded pipeline, 
- single machine map-reduce, 
- and other memory efficiency techniques.

Our results indicate that HARENS can increase the throughput of other CUDA Redundancy Elimination Systems by x7 up to speeds of 2.5Gbps.

![throughput](https://cloud.githubusercontent.com/assets/4562887/11403658/e47858c8-9352-11e5-80c4-af876147ea8b.png)

This project has been developed by [Kelu Diao](mailto:keludiao@gmail.com) and [Dr. Ioannis Papapanagiotou](mailto:ipapapa@ncsu.edu).

Requirements
-------------

- Windows 7/8 64-bit (Windows 10 possibly doesn't support CUDA architecture) or Linux 64-bit
- NVidia GPU, which supports cuda compute capability 3.5 or later
- CUDA 6.5 driver or later
- (For Windows users)
  - Visual Studio 2013 or later
  - C++ 11

User Guide
-------------

###Installation
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

Bug Reporting
-------------
* For reporting bugs please use the [HARENS/issues](https://github.com/ipapapa/HARENS/issues) page.

License
-------
© Contributors, 2015. Licensed under an [Apache-2.0](https://github.com/ipapapa/HARENS/blob/master/License) license.

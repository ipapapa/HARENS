# Hardware Accelarated Redundancy Elimination in Network Systems

[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./License)

With the tremendous growth in the amount of information stored on remote locations and cloud systems, many service providers are seeking ways to reduce the amount of redundant information sent across networks by using data de-duplication techniques. Data de-duplication can reduce network traffic without loss of information, and therefore increase available network bandwidth. However, due to heavy computations, de-duplication itself can become a bottleneck in high capacity links. Here, we propose a method named Hardware Accelerated Redundancy Elimination in Network Systems (HARENS). HARENS can significantly improve the performance of the redundancy elimination in network system by leveraging:
- the use of General Purpose Graphics Processing Unit (GPGPU) acceleration,  
- hierarchical multi-threaded pipeline, 
- single machine map-reduce, 
- and other memory efficiency techniques.

Our results indicate that HARENS can increase the throughput of other CUDA Redundancy Elimination Systems by a factor of 9 up to speeds of 3.0 Gbps. We compared our implementation with
- A naive C++ implementation of Rabin fingerprint DRE;		
- A multi-threaded accelarated algorithm;		
- A CUDA accelarated algorithm;		
- HARENS.

We used Intel Core i7-5930K 3.5GHz 12 cores CPU, 32GB DDR4 RAM, and Nvidia Tesla K40c. We used experimental data based on Youtube traces collected by Zink and publically available at the [UMass trace repository](http://traces.cs.umass.edu/index.php/Network/Network).

![throughput](https://cloud.githubusercontent.com/assets/4562887/11403658/e47858c8-9352-11e5-80c4-af876147ea8b.png)


Algorithm Overview
-------------------

- Packet/Object chunking: a sliding window to scan through the whole input stream, and marks the beginning of a window as a fingerprint based on MODP Rabin Fingerprint
- Compute a SHA-1 for chunk. We chose SHA-1 because it is light-weighted and has low has collisions.
- LRU as our chunk replacement algorithm 
- CUDA Accelaration
-- Rabin hash is computed for each window in the GPU
-- Transferring data to shared memory before computating
-- Make two copies of the data in shared memory and aligned data to avoid half of the access conflicts as well as improve memory bandwidth
-- Balanced the size of shared memory and registers allocated per multi-processor (100% theoretical occupancy and 89.91% achieved occupancy).
- Single Machine Map-Reduce
-- Launched multiple threads to execute chunk matching tasks
- Multi-threaded Pipeline to minimize device idling
- Asychronous Memory transfers

Above, we provide a summary of the techniques we have used. There are more memory efficient techniques that somebody may find in our code base.

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
  - If using Debian, type the folling command
    - ```cd \path\to\Project\HARENS```
    - ```make clean``` (only when you want to re-install)
    - ```make install```
    - ```make all```
  - If using Ubuntu
    - open the file \path\to\Project\HARENS\src\makefile with your favorite editor
    - uncommand the lines after *##commands for ubuntu*
    - command the lines after *##commands for debian*
    - ```cd \path\to\Project\HARENS```
    - ```make clean``` (only when you want to re-install)
    - ```make install```
    - ```make all```
  - If other system
    - Edit makefile (I'm sure you can handle this)
    - ```make install```
    - ```make all```
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
* We are looking forward to pull requests from everybody through [HARENS/pulls](https://github.com/ipapapa/HARENS/pulls).

License
-------
© Contributors, 2015. Licensed under an [Apache-2.0](https://github.com/ipapapa/HARENS/blob/master/License) license.
This project was initially developed by [Kelu Diao](mailto:diaokelu@gmail.com) and [Dr. Ioannis Papapanagiotou](mailto:ipapapa@ncsu.edu).


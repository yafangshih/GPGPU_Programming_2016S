This program supports two types of operation on text: <br />
* Count the Position in Words
* Find the Heads

### Example <br />
**Count the Position in Words** <br />
```
gpu qq  a hello   sonoda (input) 
123012001012345000123456 (output) 
```

**Find the Heads** <br />
```
gpu qq  a hello   sonoda (input) 
123012001012345000123456 (output from above) 
0,4,8,10,18 (output) 
```

### Build <br />
```Shell
$ nvcc -std=c++11 -arch=sm_30 -O2 -c counting.cu -o counting.o
$ nvcc -std=c++11 -arch=sm_30 -O2 main.cu counting.o -o main
```


This program supports two types of text transformation: <br />

**Convert all characters to be captial** <br />
```
$ ./main test.txt 
$ 0 
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Qu (input)
LOREM IPSUM DOLOR SIT AMET, CONSECTETUR ADIPISCING ELIT. QU (output)
```

**Swap all pairs in all words** <br />
```
$ ./main test.txt 
$ 1
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Qu (input)
oLerm piusm odolr ist mate, ocsnceetutr dapisiicgn leti. uQ (output)
```

### Build <br />
```Shell
$ nvcc main.cu -std=c++11 -o main
```

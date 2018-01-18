This program supports two types of text transformation: <br />
* Convert all characters to be captial
* Swap all pairs in all words

### Example <br />
**test.txt** <br />
> Lorem ipsum dolor sit amet, consectetur adipiscing elit. Qu <br />

**Convert all characters to be captial** <br />
```
$ ./main test.txt 
$ 0 
```
> LOREM IPSUM DOLOR SIT AMET, CONSECTETUR ADIPISCING ELIT. QU <br />

**Swap all pairs in all words** <br />
```
$ ./main test.txt 
$ 1
```
> oLerm piusm odolr ist mate, ocsnceetutr dapisiicgn leti. uQ <br />

### Build <br />
```Shell
$ nvcc main.cu -std=c++11 -o main
```

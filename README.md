# Perceptron
Natural Language Processing employes a simple perceptron to predict classes of reviews as True/Fake and Pos/Neg.
In regular machine learning applications, the perceptron uses multiple features for predictions. The NLP implementation uses word counts as features.


## Algorithm
#### Vanilla Perceptron
Training algorithm:

    wd ← 0, for all d = 1 . . . D // initialize weights
    b ← 0 // initialize bias
    for iter = 1 . . . MaxIter do
      for all (x,y) ∈ D do
        a ← ∑wd xd + b // compute activation for this example
        if ya ≤ 0 then
          wd ← wd + yxd
          for all d = 1 . . . D // update weights
          b ← b + y // update bias
        end if
     end for
    end for
    return w0, w1, . . . , wD, b
Testing Algorithm:

    a ← ∑wd xˆd + b // compute activation for the test example
    return sign(a)

#### Averaged Perceptron:
Training algorithm:

    w ← h0, 0, . . . 0i , b ← 0 // initialize weights and bias
    u ← h0, 0, . . . 0i , β ← 0 // initialize cached weights and bias
    c ← 1 // initialize example counter to one
    for iter = 1 . . . MaxIter do
      for all (x,y) ∈ D do
        if y(w · x + b) ≤ 0 then
          w ← w + y x // update weights
          b ← b + y // update bias
          u ← u + y c x // update cached weights
          β ← β + y c // update cached bias
        end if
      c ← c + 1 // increment counter regardless of update
      end for
    end for
    return w - u/c, b - β/c
    
## Usage

    python3 perceplearn.py <inputfile-traindata>
    python3 percepclassify.py <modelfile> <inputfile-testdata>
    python3 percepevaluate.py <inputfile-testkeys>
 
modelfile : vanillamodel.txt | averagedmodel.txt

## Results
f1 scores against given dev-test dataset

- Vanilla Model: 89.03
- Averaged Model: 89.05
 

## Neural network decision boundary visualisation

This is a naive demo of multi-class and multi-label classification.
The demo allows you to alter the weights of 3 classifiers, one for each class (blueberries, bananas, strawberries).
The weights vectors are visualised using arrows, and the decision boundaries by using lines.
Moving the weight vectors around, changes the decision boundaries.
It also changes the probability of each input point belonging to one of the three classes, as visualised on the 3d plot on the right.


## How to move things

The 3d plot can be moved around and the 2d plot can be zoomed in/out by scrolling.
Choose which class you want to change the weight vector for from the radio buttons and click on the left plot to change its weights.
You can also alter the bias term for each class by clicking on the bars at the top right.

If sigmoids is chosen, then this is interpreted as a multi-label classification setting and there is an independent binary classifier fit for each class.
In this case therefore, there is a separate probability for each class being true, and we can visualise these three probabilities as
a (x, y, z) point constrained to be in the unit cube (as seen below on the right).

![Decision boundary with intersecting lines and sigmoids](http://grv.overfit.xyz/static/images/lines.jpg)

If softmax is chosen, then classes are mutually exclusive (the score for each class is normalized to form a probability distribution over all classes).
Therefore, the probability of each class for each input point is constrained to be a (x, y, z) point on the 2-simplex (as seen below).
![Decision boundary with intersecting lines and softmax](http://grv.overfit.xyz/static/images/softmax.jpg)

If we also choose circle from the first radio button, we augment the input using basis functions so that we get circular decision boundaries.
![Decision boundary with intersecting circles and sigmoids](http://grv.overfit.xyz/static/images/circles.jpg)


### Requirements

* Python3.7 (should work on older python3)
* matplotlib
* numpy


### How to run

``` bash
git clone https://github.com/andreasgrv/decision_boundary_demo
cd decision_boundary_demo
python3.7 -m venv .env && source .env/bin/activate
pip install -r requirements.txt
python main.py
```

### Options

* Linear or circular decision boundaries
* Choose weights and biases
* Sigmoids or softmax


### Notes

* For softmax the decision boundaries are formed by the differences of weight vectors, that is not currently mirrored in the lines but should be mirrored in the colours of the points.

## Neural network decision boundary visualisation

This is a simple demo of 3-class classification using neural networks.
Since the input is unrealistic, the classes are also unrealistic and classnames are used for color coding purposes (Fruit are a legend.).
The visualisation shows the decision boundaries in 2d input space and how they map points in the input to 3d probability space.

* If sigmoids is chosen, then this is a multi label classification setting and the probability space is the unit cube.
* If softmax is chosen, then this is the multi class setting and the probability space is the 2-simplex.

The 3d plot can be moved around and the 2d plot can be zoomed in/out by scrolling.
Choose what class you want to change the decision boundary for from the radio buttons and click on the left plot to change its weights.

## Snapshots
![Decision boundary with intersecting lines and sigmoids](http://grv.overfit.xyz/lines.jpg)
![Decision boundary with intersecting lines and softmax](http://grv.overfit.xyz/softmax.jpg)
![Decision boundary with intersecting circles and sigmoids](http://grv.overfit.xyz/circles.jpg)


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

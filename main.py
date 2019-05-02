from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
import matplotlib.pyplot as plt


# https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2018-9/master/mlp/layers.py
def softmax(inputs):
    exp_inputs = np.exp(inputs - inputs.max(-1)[:, None])
    return exp_inputs / exp_inputs.sum(-1)[:, None]


def sigmoid(inputs):
     return 1. / (1. + np.exp(-inputs))


def plot_decision_boundaries(ax, ws, bs, colors):
    global function
    lines, arrows = [], []
    if function == 'linear':
        # ws = weights, bs = biases
        # decision boundary : y = ax + b
        # line is perpendicular to weight vector
        # Calculate the slope a
        a = - (ws[:, 0] / ws[:, 1])
        # Calculate the offset b
        offset = - bs / ws[:, 1]
        
        # Create two points to plot line
        points = np.array([-1000, 1000]).reshape(1, -1)

        a = a.reshape(-1, 1)
        offset = offset.reshape(-1, 1)

        dec_bound_points = a * points + offset

        for i, ys in enumerate(dec_bound_points):
            ww = np.array(ws[i, :])
            start = (0, 0)
            arrows.append(ax.annotate('', xy=ww, xytext=start,
                                      arrowprops=dict(arrowstyle="->", color=colors[i]),
                                      ha='left', va='top',
                                      color=colors[i]))
            # This adds the label (hard to place text in front of arrow otherwise)
            arrows.append(ax.annotate(choices[i], xy=start, xytext=ww,
                                      ha='left', va='top',
                                      color=colors[i]))
            lines.append(ax.plot(points.reshape(-1), ys, label=choices[i], c=colors[i]))
    else:
        # plot circular decision boundary
        for i, ww in enumerate(ws):
            crcl = plt.Circle(ww, np.exp(bs[i]), color=colors[i], fill=False, label=choices[i])
            ax.plot(*ww, 'o', color=colors[i])
            ax.add_patch(crcl)
            lines.append(crcl)
    return lines, arrows


def change_w(event):
    if str(event.inaxes).startswith('AxesSubplot'):
        global weights, bias, w_index
        weights[w_index, 0] = np.float32(event.xdata)
        weights[w_index, 1] = np.float32(event.ydata)
        clear_plot()
        plot(weights, bias)
        plt.draw()


# stolen from https://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel
def zoom_fun(event, base_scale=2.):
    global X_HIGH, X_LOW
    if event.button == 'up':
        # deal with zoom in
        scale_factor = 1/base_scale
    elif event.button == 'down':
        # deal with zoom out
        scale_factor = base_scale
    X_HIGH *= scale_factor
    X_LOW = - X_HIGH
    clear_plot()
    plot(weights, bias)
    plt.draw()


def plot(weights, bias):
    global ax, ax2, function, w_index
    if function == 'linear':
        act = eucl_points.dot(weights.T) + bias
    else:
        points = np.expand_dims(eucl_points, 1)
        distances = np.sqrt(np.sum((points - weights) ** 2, axis=2))
        act = np.exp(bias) - distances
    if activation == 'softmax':
        act = softmax(act)
        decisions = np.zeros_like(act)
        decision_idxs = np.argmax(act, axis=1)
        decisions[np.arange(len(decision_idxs)), decision_idxs] = 1.
    else:
        decisions = np.array((act > .0), dtype=np.int8)
        act = sigmoid(act)
    eucl_colors = decisions.dot(colors.reshape(3, 4))/(np.sum(decisions, axis=1, keepdims=True) + 1e-5)
    eucl_colors[:, -1] = 1.

    lines, arrows = plot_decision_boundaries(ax2, weights, bias, colors)
    epoints = ax2.scatter(*eucl_points.T, s=2, c=eucl_colors)
    vpoints = ax.scatter(*act.T, c=eucl_colors)

    ax2.grid(True)
    ax2.set_xlim([X_LOW, X_HIGH])
    ax2.set_ylim([X_LOW, X_HIGH])
    ax2.set_xlabel('Click to change decision boundary for %s' % choices[w_index])
    ax.set_xlabel('%s probability' % choices[0])
    ax.set_ylabel('%s probability' % choices[1])
    ax.set_zlabel('%s probability' % choices[2])
    return epoints, vpoints, lines


def clear_plot():
    global ax, ax2
    ax.cla()
    ax2.cla()


def choose_w(label):
    global w_index
    mapper = dict(zip(choices, range(len(choices))))
    idx = mapper[label]
    w_index = idx


def update_b1(val):
    global bias
    bias[0] = val
    clear_plot()
    plot(weights, bias)
    plt.draw()


def update_b2(val):
    global bias
    bias[1] = val
    clear_plot()
    plot(weights, bias)
    plt.draw()


def update_b3(val):
    global bias
    bias[2] = val
    clear_plot()
    plot(weights, bias)
    plt.draw()


def choose_nonlinearity(label):
    global activation
    activation = label
    clear_plot()
    plot(weights, bias)
    plt.draw()


def choose_function(label):
    global function
    function = label
    clear_plot()
    plot(weights, bias)
    plt.draw()


choices = ['blueberries', 'bananas', 'strawberries']
activation = 'sigmoids'
function = 'linear'
X_HIGH = 5
X_LOW = - X_HIGH


if __name__ == "__main__":
    weights = np.array([[1., 1.], [-1, 1.], [1., -.5]])
    bias = np.array([0., 0., 0])

    w_index = 0
    NUM_POINTS = 1000

    eucl_points = np.random.uniform(-4, 4, (NUM_POINTS, 2))
    colors = plt.cm.tab10([0, 8, 3])

    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax2 = fig.add_subplot(1, 2, 1, aspect='equal')
    plt.subplots_adjust(left=0.12)
    rax = plt.axes([0.22, 0.9, 0.12, 0.1])
    wradio = RadioButtons(rax, choices)
    wradio.on_clicked(choose_w)

    rax2 = plt.axes([0.1, 0.9, 0.12, 0.1])
    nradio = RadioButtons(rax2, ['sigmoids', 'softmax'])
    nradio.on_clicked(choose_nonlinearity)

    rax3 = plt.axes([0., 0.9, 0.1, 0.1])
    fradio = RadioButtons(rax3, ['linear', 'circle'])
    fradio.on_clicked(choose_function)

    epoints, vpoints, lines = plot(weights, bias)
    b1rax = plt.axes([0.5, 0.96, 0.3, 0.03])
    b2rax = plt.axes([0.5, 0.93, 0.3, 0.03])
    b3rax = plt.axes([0.5, 0.9, 0.3, 0.03])
    b1 = Slider(b1rax, '%s bias' % choices[0], -10, 10, valinit=bias[0])
    b2 = Slider(b2rax, '%s bias' % choices[1], -10, 10, valinit=bias[1])
    b3 = Slider(b3rax, '%s bias' % choices[2], -10, 10, valinit=bias[2])

    b1.on_changed(update_b1)
    b2.on_changed(update_b2)
    b3.on_changed(update_b3)

    fig.canvas.callbacks.connect('button_press_event', change_w)
    fig.canvas.mpl_connect('scroll_event', zoom_fun)
    plt.show()

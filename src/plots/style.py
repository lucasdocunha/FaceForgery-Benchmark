import matplotlib.pyplot as plt
MODEL_COLORS={"resnet":"#4477AA","xception":"#EE6677","mobilenet":"#228833","vit":"#CCBB44","clip":"#66CCEE","dino":"#AA3377"}
def apply_style():
    plt.rcParams.update({"font.family":"serif","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":120})

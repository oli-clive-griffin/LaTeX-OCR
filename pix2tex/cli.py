from pix2tex.dataset.transforms import test_transform
import pandas.io.clipboard as clipboard
from PIL import ImageGrab, Image, ImageTk
import os
import io
from typing import Tuple
from contextlib import suppress
import logging
import yaml
import urllib.parse
import webbrowser
import numpy as np
import torch
from munch import Munch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from pix2tex.models import get_model
from pix2tex.utils import *
from pix2tex.model.checkpoints.get_latest_checkpoint import download_checkpoints
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def render_latex(latex_code):
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, f"${latex_code}$", horizontalalignment='center', verticalalignment='center', fontsize=20, fontname='Latin Modern Math')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    plt.close(fig)

    latex_image = Image.open(buf)
    return latex_image

def show_comparison(original_img, latex_code):

    # Render the LaTeX image
    latex_img = render_latex(latex_code)

    # Display the images using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display LaTeX rendered image
    axes[1].imshow(latex_img)
    axes[1].set_title('LaTeX Rendered Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def minmax_size(img: Image.Image, max_dimensions: Tuple[int, int] | None = None, min_dimensions: Tuple[int, int] | None = None) -> Image.Image:
    """Resize or pad an image to fit into given dimensions

    Args:
        img (Image): Image to scale up/down.
        max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
        min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

    Returns:
        Image: Image with correct dimensionality
    """
    if max_dimensions is not None:
        ratios = [a/b for a, b in zip(img.size, max_dimensions)]
        if any([r > 1 for r in ratios]):
            size = np.array(img.size)//max(ratios)
            img = img.resize(size.astype(int), Image.BILINEAR)
    if min_dimensions is not None:
        # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
        padded_size = [max(img_dim, min_dim) for img_dim, min_dim in zip(img.size, min_dimensions)]
        if padded_size != list(img.size):  # assert hypothesis
            padded_im = Image.new('L', padded_size, 255)
            padded_im.paste(img, img.getbbox())
            img = padded_im
    return img


class LatexOCR:
    '''Get a prediction of an image in the easiest way'''

    image_resizer = None

    @in_model_path()
    def __init__(self, arguments=None):
        """Initialize a LatexOCR model

        Args:
            arguments (Union[Namespace, Munch], optional): Special model parameters. Defaults to None.
        """
        if arguments is None:
            arguments = Munch({'config': 'settings/config.yaml', 'checkpoint': 'checkpoints/weights.pth', 'no_cuda': True, 'no_resize': False})
        logging.getLogger().setLevel(logging.FATAL)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        with open(arguments.config, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.args = parse_args(Munch(params))
        self.args.update(**vars(arguments))
        self.args.wandb = False
        self.args.device = 'cuda' if torch.cuda.is_available() and not self.args.no_cuda else 'cpu'
        if not os.path.exists(self.args.checkpoint):
            download_checkpoints()
        self.model = get_model(self.args)
        self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device))
        self.model.eval()

        if 'image_resizer.pth' in os.listdir(os.path.dirname(self.args.checkpoint)) and not arguments.no_resize:
            self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.args.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
                                          preact=True, stem_type='same', conv_layer=StdConv2dSame).to(self.args.device)
            self.image_resizer.load_state_dict(torch.load(os.path.join(os.path.dirname(self.args.checkpoint), 'image_resizer.pth'), map_location=self.args.device))
            self.image_resizer.eval()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.args.tokenizer)

    @in_model_path()
    def __call__(self, img, resize=True) -> str | None:
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if img is None:
            return None
        img = minmax_size(pad(img), self.args.max_dimensions, self.args.min_dimensions)
        if (self.image_resizer is not None and not self.args.no_resize) and resize:
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    img = pad(minmax_size(input_image.resize((w, h), Image.Resampling.BILINEAR if r > 1 else Image.Resampling.LANCZOS), self.args.max_dimensions, self.args.min_dimensions))
                    t = test_transform(image=np.array(img.convert('RGB')))['image'][:1].unsqueeze(0)
                    w = (self.image_resizer(t.to(self.args.device)).argmax(-1).item()+1)*32
                    logging.info(r, img.size, (w, int(input_image.size[1]*r)))
                    if (w == img.size[0]):
                        break
                    r = w/img.size[0]
        else:
            img = np.array(pad(img).convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.args.device)

        dec = self.model.generate(im.to(self.args.device), temperature=self.args.get('temperature', .25))
        pred = post_process(token2str(dec, self.tokenizer)[0])
        try:
            clipboard.copy(pred)
        except:
            pass
        return pred


# BASE_DATA = '{"displayMode":true,"leqno":false,"fleqn":false,"throwOnError":true,"errorColor":"#cc0000","strict":"warn","output":"htmlAndMathml","trust":false,"code":"%s"}'
# def render_as_web(pred):
#     data = BASE_DATA % pred.replace('\\', '\\\\')
#     url = 'https://katex.org/?data=' + urllib.parse.quote(data)
#     webbrowser.open(url)

def main(arguments):
    model = LatexOCR(arguments)
    image = ImageGrab.grabclipboard()
    if image is None:
        print('No image in clipboard')
        return
    pred_latex = model(image)
    show_comparison(image, pred_latex)

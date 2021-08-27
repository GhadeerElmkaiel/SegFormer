from argparse import ArgumentParser
import os
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import mmcv
from PIL import Image
import numpy as np
import time



def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file') # default='local_configs/segformer/B2/segformer.b2.1024x1024.sber.160k.py'
    parser.add_argument('checkpoint', help='Checkpoint file') #  default='work_dirs/segformer.b2.1024x1024.sber.160k/iter_160000.pth'
    parser.add_argument('--images', help='Images path', default='/home/ghadeer/Projects/Datasets/SberMerged/test/images/')
    parser.add_argument('--save_path', help='Path to save resulted images', default='results/')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='sber',
        help='Color palette used for segmentation map')
    args = parser.parse_args()
    if args.save_path == "results/":
        save_path = "results/" + str(time.time()) + '/'
    else:
        save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # Create a list of all images:
    images = [x for x in os.listdir(args.images) if "." in x]
    palette = [[102, 255, 102], [51, 221, 255], [245, 147, 49], [184, 61, 245], [250, 50, 83], [0, 0, 0]]
    palette = np.array(palette)

    for name in images:
        path_to_img = args.images + name

        # Getting the results from the model
        result, output = inference_segmentor(model, path_to_img)
        seg = result[0]

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

        # Recolor the resulted image to match the needed colors
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        
        color_seg = color_seg[..., ::-1]
        img = color_seg.astype(np.uint8)


        # Saving the resulted image
        image = Image.fromarray(mmcv.bgr2rgb(img))
        image.save(save_path+name)


if __name__ == '__main__':
    main()

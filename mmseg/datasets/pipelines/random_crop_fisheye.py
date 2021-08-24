import numpy as np

from ..builder import PIPELINES
from IPython import embed

@PIPELINES.register_module()
class RandomFisheyeCrop(object):
    """Random crop the image & seg with fisheye bounds.

        Used for random augmentation of fisheye images
    """

    def __init__(self,  cat_max_ratio=1., ignore_index=255, mvx = 100, const_crop_y = 150, rand_crop_y = 50, crop_prob = 0.5, shift_prob = 0.5):
        """RandomFisheyeCrop constructor

        Args:
            cat_max_ratio (float, optional): The maximum ratio that single category could
            occupy. Defaults to 1.0.
            ignore_index (int, optional): [description]. Defaults to 255.
            mvx (int, optional): Max shift over the X axis in pixels. Defaults to 100.
            const_crop_y (int, optional): Constant crop over the Y axis. Defaults to 150.
            rand_crop_y (int, optional): Random crop over the Y axis. Defaults to 50.
            crop_proba (float, optional): Crop probability. Defaults to 0.5.
            shift_proba (float, optional): Shift probability. Defaults to 0.5.
        """
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        self.mvx = mvx#100
        self.const_crop_y = const_crop_y #150
        self.rand_crop_y = rand_crop_y #50
        self.crop_prob = crop_prob
        self.shift_prob = shift_prob
        

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        H,W = img.shape[:2]
        rand_crop_top = np.random.randint(self.rand_crop_y)
        rand_crop_bot = self.rand_crop_y - rand_crop_top

        crop_y1 = self.const_crop_y+rand_crop_top
        crop_y2 = H-self.const_crop_y-rand_crop_bot

        return crop_y1, crop_y2

    def crop(self, img, crop_bbox, is_mask = False):
        """Crop from ``img``"""
        if(self.cropped):
            crop_y1, crop_y2 = crop_bbox
            img = img[crop_y1:crop_y2, :,:]
            self.crop_margins = crop_y1, crop_y2
        else:
            self.crop_margins = 0, img.shape[0]
        
        if(self.shifted):
            rand_mvx = np.random.randint(self.mvx)
            if(is_mask):
                stack_line = 255*np.ones((img.shape[0], rand_mvx,  3))
            else:
                stack_line = np.zeros((img.shape[0], rand_mvx,  3))
            img = np.hstack([img[:, rand_mvx:], stack_line.astype(img.dtype)])
            self.shift_dist = rand_mvx
        else:
            self.shift_dist = 0
            
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        self.shifted = np.random.rand() < self.shift_prob
        self.cropped = np.random.rand() < self.crop_prob
        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['shifted'] = self.shifted
        results['cropped'] = self.cropped
        results['shift'] = self.shift_dist
        results['crop_margins'] = self.crop_margins

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox, is_mask=True)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

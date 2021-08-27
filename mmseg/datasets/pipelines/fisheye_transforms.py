import cv2
import numpy as np
from ..builder import PIPELINES

@PIPELINES.register_module()
class RandomFisheyeCrop(object):
    """Random crop the image & seg with fisheye bounds.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self,  bbox = None, prob = 1.0, part_x = 1.0, part_y = 1.0, max_dx = 0, max_dy = 0, part_x_range = None, part_y_range = None, dx_range = None, dy_range = None):
        if not bbox is None:
            bbox_W = bbox[1][0] - bbox[0][0] + 1
            bbox_H = bbox[1][1] - bbox[0][1] + 1
            if bbox_W < 2 or bbox_H <2:
                raise  AssertionError("BBox has to be greater than 1 in width and height. It also should have the form of [[Top Left X, Top Left Y],[Right Bottom X, Right Bottom Y]]")
        self.bbox = bbox
        if prob < 0 or prob > 1:
            raise AssertionError("Probability has to be in range [0.0, 1.0]")
        self.prob = prob
        # Set dx_range
        if not dx_range is None:
            if not isinstance(dx_range,list) and not isinstance(dx_range, tuple) and not isinstance(dx_range,set) or len(dx_range)!=2:
                raise AssertionError("dx_range should be list, tuple or set with lenght 2")
            self.dx_range = [np.round(min(list(map(abs, dx_range)))).astype('int'), np.round(max(list(map(abs, dx_range)))).astype('int')+1]
        elif not max_dx is None:
            self.dx_range = [0, np.round(abs(max_dx)).astype('int')+1]
        else:
            self.dx_range = [0,1]
        
        # Set dy range
        if not dy_range is None:
            if not isinstance(dy_range,list) and not isinstance(dy_range, tuple) and not isinstance(dy_range,set) or len(dy_range)!=2:
                raise AssertionError("dy_range should be list, tuple or set with lenght 2")
            self.dy_range = [np.round(min(list(map(abs, dy_range)))).astype('int'), np.round(max(list(map(abs, dy_range)))).astype('int')+1]
        elif not max_dy is None:
            self.dy_range = [0, np.round(abs(max_dy)).astype('int')+1]
        else:
            self.dy_range = [0,1]

        # Set part_x_range
        if not part_x_range is None:
            if not isinstance(part_x_range,list) and not isinstance(part_x_range, tuple) and not isinstance(part_x_range,set) or len(part_x_range)!=2:
                raise AssertionError("part_x_range should be list, tuple or set with lenght 2")
            if part_x_range[0] <= 0 or part_x_range[1] <= 0:
                raise AssertionError("part of image should be greater then 0")
            self.part_x_range = part_x_range
        elif not part_x is None:
            if part_x <= 0:
                raise AssertionError("part_x has to be greater than 0")
            self.part_x_range = [1e-3, part_x]
        else:
            raise AssertionError("There should be at least on of part_x or part_x_range")
        # Set part_y_range
        if not part_y_range is None:
            if not isinstance(part_y_range,list) and not isinstance(part_y_range, tuple) and not isinstance(part_y_range,set) or len(part_y_range)!=2:
                raise AssertionError("part_y_range should be list, tuple or set with lenght 2")
            if part_y_range[0] <= 0 or part_y_range[1] <= 0:
                raise AssertionError("part of image should be greater then 0")
            self.part_y_range = part_y_range
        elif not part_y is None:
            if part_y <= 0:
                raise AssertionError("part_y has to be greater than 0")
            self.part_y_range = [1e-3, part_y]
        else:
            raise AssertionError("There should be at least on of part_y or part_y_range")
        
        
    def get_crop_box(self,pic_size):
        H,W = pic_size
        bbox_w = self.bbox[1][0]-self.bbox[0][0]+1
        bbox_h = self.bbox[1][1]-self.bbox[0][1]+1
        bbox_center = [self.bbox[0][0] + bbox_w*0.5, self.bbox[0][1] + bbox_h*0.5]
        self.px_m = np.random.uniform(*self.part_x_range)
        self.py_m = np.random.uniform(*self.part_y_range)
        dim_x = np.ceil(bbox_w*self.px_m).astype('int')
        dim_y = np.ceil(bbox_h*self.py_m).astype('int')
        crop_left = np.ceil(bbox_center[0] - 0.5*dim_x).astype('int')
        crop_right = np.floor(bbox_center[0] + 0.5*dim_x).astype('int')
        crop_top = np.ceil(bbox_center[1] - 0.5*dim_y).astype('int')
        crop_bottom = np.floor(bbox_center[1] + 0.5*dim_y).astype('int')
        if(self.dx_range[1]-self.dx_range[0] > 1):
            dx = pow(-1, np.random.randint(2)) * np.random.randint(*self.dx_range)
            crop_left += dx
            crop_right += dx
        else:
            dx = 0
        if(self.dy_range[1]-self.dy_range[0] > 1):
            dy = pow(-1, np.random.randint(2)) * np.random.randint(*self.dy_range)
            crop_top += dy
            crop_bottom += dy
        else:
            dy = 0
        if self.px_m < 1:
            new_bbox_left = max(-dx,0)
            new_bbox_right = min(-dx + dim_x, dim_x - 1)
        else:
            new_bbox_left = max(np.ceil((dim_x - bbox_w)*0.5).astype('int') - dx, 0)
            new_bbox_right = min(np.ceil((dim_x - bbox_w)*0.5).astype('int') - dx + bbox_w, dim_x -1)
        if self.py_m < 1:
            new_bbox_top = max(-dy,0)
            new_bbox_bottom = min(-dy + dim_y, dim_y - 1)
        else:
            new_bbox_top = max(np.ceil((dim_y - bbox_h)*0.5).astype('int') - dy, 0)
            new_bbox_bottom = min(np.ceil((dim_y - bbox_h)*0.5).astype('int') - dy + bbox_h, dim_y -1)
        if crop_left < 0:
            pad_left = crop_left
            crop_left = 0
        else:
            pad_left = 0
        if crop_right > W - 1:
            pad_right = crop_right - W - 1
            crop_right = W - 1
        else:
            pad_right = 0
        if crop_top < 0:
            pad_top = crop_top
            crop_top = 0
        else:
            pad_top = 0
        if crop_bottom > H - 1:
            pad_bottom = crop_bottom - H + 1
            crop_bottom = H - 1
        else:
            pad_bottom = 0
        crop = (crop_top,crop_bottom,crop_left,crop_right)
        pad = (abs(pad_top), abs(pad_bottom), abs(pad_left), abs(pad_right))
        new_bbox = [[new_bbox_left, new_bbox_top],[new_bbox_right, new_bbox_bottom]]
        return crop, pad, new_bbox


            
             
    
    def crop(self, pic, crop, pad, is_mask = False):
        color = [255*is_mask, 255*is_mask, 255*is_mask]
        cropped = cv2.copyMakeBorder(pic[crop[0]:crop[1],crop[2]:crop[3],:], *pad, cv2.BORDER_CONSTANT, value = color)
        return cropped
    

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        self.cropped = np.random.rand()<self.prob
        if(self.cropped):
            if('fisheye_bbox' in results.keys()):
                self.bbox = results['fisheye_bbox']
            elif self.bbox is None:
                raise AssertionError("There is no information abot bbox. You should to define it while init this transform or in the previous transform's results")
            img = results['img']
            crop, pad, new_bbox = self.get_crop_box(img.shape[0:2])
            results['img'] = self.crop(img, crop, pad)
            for key in results.get('seg_fields', []):
                results[key] = self.crop(results[key], crop, pad, is_mask=True)
            results['fisheye_bbox'] = new_bbox
            results['img_shape'] = results['img'].shape
        results['cropped'] = self.cropped
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(part_x_range={self.part_x_range} | part_y_range={self.part_y_range})'
        


@PIPELINES.register_module()
class RandomFisheyeShift(object):
    """Random shift the image & seg with fisheye bounds.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self,  bbox = None, prob = 1.0, max_dx = 0, max_dy = 0, dx_range = None, dy_range = None):
        if not bbox is None:
            bbox_W = bbox[1][0] - bbox[0][0] + 1
            bbox_H = bbox[1][1] - bbox[0][1] + 1
            if bbox_W < 2 or bbox_H <2:
                raise  AssertionError("BBox has to be greater than 1 in width and height. It also should have the form of [[Top Left X, Top Left Y],[Right Bottom X, Right Bottom Y]]")
        self.bbox = bbox
        if prob < 0 or prob > 1:
            raise AssertionError("Probability has to be in range [0.0, 1.0]")
        else:
            self.prob = prob
        # Set dx range
        if not dx_range is None:
            if not isinstance(dx_range,list) and not isinstance(dx_range, tuple) and not isinstance(dx_range,set) or len(dx_range)!=2:
                raise AssertionError("dx_range should be list, tuple or set with lenght 2")
            self.dx_range = [np.round(min(list(map(abs, dx_range)))).astype('int'), np.round(max(list(map(abs, dx_range)))).astype('int')+1]
        elif not max_dx is None:
            self.dx_range = [0, np.round(abs(max_dx)).astype('int')+1]
        else:
            raise AssertionError("There should be at least on of max_dx or dx_range")
        # Set dy range
        if not dy_range is None:
            if not isinstance(dy_range,list) and not isinstance(dy_range, tuple) and not isinstance(dy_range,set) or len(dy_range)!=2:
                raise AssertionError("dy_range should be list, tuple or set with lenght 2")
            self.dy_range = [np.round(min(list(map(abs, dy_range)))).astype('int'), np.round(max(list(map(abs, dy_range)))).astype('int')+1]
        elif not max_dy is None:
            self.dy_range = [0, np.round(abs(max_dy)).astype('int')+1]
        else:
            raise AssertionError("There should be at least on of max_dy or dy_range")
        
        
    def get_shift_values(self,pic_size):
        H,W = pic_size
        # Get dx
        if self.dx_range[1] - self.dx_range[0] > 1:
            dx = pow(-1, np.random.randint(2)) * np.random.randint(*self.dx_range)
            dx = max(dx, -self.bbox[0][0]) if dx <=0 else min(dx, W-self.bbox[1][0]-1)
        else:
            dx = 0
        # Get dy
        if self.dy_range[1] - self.dy_range[0] > 1:
            dy = pow(-1, np.random.randint(2)) * np.random.randint(*self.dy_range)
            dy = max(dy, -self.bbox[0][1]) if dy <=0 else min(dy, H-self.bbox[1][1]-1)
        else:
            dy = 0
        return dx, dy
        
    
    def shift(self, dx, dy, pic, is_mask = False):
        H,W = pic.shape[:-1]
        color = is_mask*255
        if(dx<0):
            pic = pic[ :,abs(dx):, :]
            pic = np.hstack([pic, color*np.ones(( pic.shape[0],abs(dx), 3), dtype = pic.dtype)])
            if(not is_mask):
                self.bbox[0][0]+=dx
                self.bbox[1][0]+=dx
        elif(dx>0):
            pic = pic[ :,:W-abs(dx), :]
            pic = np.hstack([color*np.ones(( pic.shape[0],abs(dx), 3), dtype = pic.dtype),pic])
            if(not is_mask):
                self.bbox[0][0]+=dx
                self.bbox[1][0]+=dx
        if(dy<0):
            pic = pic[abs(dy):,:,  :]
            pic = np.vstack([pic, color*np.ones((abs(dy),pic.shape[1],  3), dtype = pic.dtype)])
            if(not is_mask):
                self.bbox[0][1]+=dy
                self.bbox[1][1]+=dy
        elif(dy>0):
            pic = pic[:H-abs(dy),:,  :]
            pic = np.vstack([color*np.ones((abs(dy),pic.shape[1],  3), dtype = pic.dtype),pic])
            if(not is_mask):
                self.bbox[0][1]+=dy
                self.bbox[1][1]+=dy
        return pic
    

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        self.shifted = np.random.rand()<self.prob
        if(self.shifted):
            if('fisheye_bbox' in results.keys()):
                self.bbox = results['fisheye_bbox']
            elif self.bbox is None:
                raise AssertionError("There is no information abot bbox. You should to define it while init this transform or in the previous transform's results")
            img = results['img']
            dx, dy = self.get_shift_values(img.shape[0:2])
            results['img'] = self.shift(dx, dy, img)
            for key in results.get('seg_fields', []):
                results[key] = self.shift(dx, dy, results[key], is_mask=True)
            results['dx'] = dx
            results['dy'] = dy
            results['fisheye_bbox'] = self.bbox
        results['shifted'] = self.shifted
       
        
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.size})'
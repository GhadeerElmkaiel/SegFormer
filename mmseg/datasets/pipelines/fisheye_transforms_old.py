import numpy as np
import cv2

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomFisheyeShift(object):
    """Random shift the image & seg with fisheye bounds.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self,  bbox = None, prob = 1.0, max_dx = 10, max_dy = 10, dx_range = None, dy_range = None):
        self.bbox = bbox
        self.prob = prob
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.dx_range = dx_range
        self.dy_range = dy_range
        
        self.dx = 0
        self.dy = 0
        
        
    def get_shift_values(self,pic_size):
        H,W = pic_size
        
        if(self.dx_range and self.dy_range):
            dx_range_left = [0,0]
            dy_range_top = [0,0]
            dx_range_right = [0,0]
            dy_range_bot = [0,0]
            dx_range_left[0] = min(self.dx_range[0], self.bbox[0][0]-1)
            dx_range_left[1] = min(self.dx_range[1], self.bbox[0][0])
            dx_range_right[0] = min(self.dx_range[0], W-self.bbox[1][0]-1)
            dx_range_right[1] = min(self.dx_range[1], W-self.bbox[1][0])
            dy_range_top[0] = min(self.dy_range[0], self.bbox[0][1]-1)
            dy_range_top[1] = min(self.dy_range[1], self.bbox[0][1])
            dy_range_bot[0] = min(self.dy_range[0], H-self.bbox[1][1]-1)
            dy_range_bot[1] = min(self.dy_range[1], H-self.bbox[1][1])
            dx = [-np.random.randint(dx_range_left[0], dx_range_left[1]), np.random.randint(dx_range_right[0], dx_range_right[1])]
            self.dx = dx[np.random.randint(2)]
            dy = [-np.random.randint(dy_range_top[0], dy_range_top[1]), np.random.randint(dy_range_bot[0], dy_range_bot[1])]
            self.dy = dy[np.random.randint(2)]
        else:
            dx_range_left = min(self.max_dx, self.bbox[0][0])
            dx_range_right = min(self.max_dx, W-self.bbox[1][0])
            dy_range_top = min(self.max_dy, self.bbox[0][1])
            dy_range_bot = min(self.max_dy, H-self.bbox[1][1])
            self.dx = np.random.randint(-dx_range_left, dx_range_right)
            self.dy = np.random.randint(-dy_range_top, dy_range_bot)
    
    def shift(self,pic, is_mask = False):
        H,W = pic.shape[:-1]
        color = is_mask*255
        if(self.dx<0):
            pic = pic[ :,abs(self.dx):, :]
            pic = np.hstack([pic, color*np.ones(( pic.shape[0],abs(self.dx), 3), dtype = pic.dtype)])
            self.bbox[0][0]+=self.dx
            self.bbox[1][0]+=self.dx
        elif(self.dx>0):
            pic = pic[ :,:W-abs(self.dx), :]
            pic = np.hstack([color*np.ones(( pic.shape[0],abs(self.dx), 3), dtype = pic.dtype),pic])
            self.bbox[0][0]+=self.dx
            self.bbox[1][0]+=self.dx
        if(self.dy<0):
            pic = pic[abs(self.dy):,:,  :]
            pic = np.vstack([pic, color*np.ones((abs(self.dy),pic.shape[1],  3), dtype = pic.dtype)])
            self.bbox[0][1]+=self.dy
            self.bbox[1][1]+=self.dy
        elif(self.dy>0):
            pic = pic[:H-abs(self.dy),:,  :]
            pic = np.vstack([color*np.ones((abs(self.dy),pic.shape[1],  3), dtype = pic.dtype),pic])
            self.bbox[0][1]+=self.dy
            self.bbox[1][1]+=self.dy
        return pic
    

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        if('fisheye_bbox' in results.keys()):
            self.bbox = results['fisheye_bbox']
        img = results['img']
        self.shifted = np.random.rand()<self.prob
        if(self.shifted):
            self.get_shift_values(img.shape[0:2])
            results['img'] = self.shift(img)
            for key in results.get('seg_fields', []):
                results[key] = self.shift(results[key], is_mask=True)
        results['dx'] = self.dx
        results['dy'] = self.dy
        results['shifted'] = self.shifted
        results['fisheye_bbox'] = self.bbox
        
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'
        
        
        
@PIPELINES.register_module()
class RandomFisheyeCrop(object):
    """Random crop the image & seg with fisheye bounds.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self,  bbox = None, size = None, keep_ratio = False,prob = 1.0, part_x = 1.0, part_y = 1.0, max_dx = 1, max_dy = 1, part_x_range = None, part_y_range = None, dx_range = None, dy_range = None):
        self.bbox = bbox
        self.bbox_center = [(bbox[0][0]+bbox[1][0])/2.0, (bbox[0][1]+bbox[1][1])/2.0]
        self.dx_orig = (bbox[1][0]-bbox[0][0])/2.0
        self.dy_orig = (bbox[1][1]-bbox[0][1])/2.0
        self.px_m = 0
        self.py_m = 0
        self.dx_delta = 0
        self.dy_delta = 0
        self.prob = prob
        self.size = size
        self.max_dx = max_dx
        self.max_dy = max_dy
        self.dx_range = dx_range
        self.dy_range = dy_range
        self.part_x = part_x
        self.part_y = part_y
        self.part_x_range = part_x_range
        self.part_y_range = part_y_range
        self.keep_ratio = keep_ratio
        
        self.dx = 0
        self.dy = 0
        
        
    def get_crop_box(self,pic_size):
        H,W = pic_size
        if(self.part_x_range and self.part_y_range):
            self.px_m = np.random.uniform(self.part_x_range[0], self.part_x_range[1])
            self.py_m = np.random.uniform(self.part_y_range[0], self.part_y_range[1])
            dim_x = abs(int(self.dx_orig*self.px_m))
            dim_y = abs(int(self.dy_orig*self.py_m))
            scaled_bbox = [self.bbox_center[0]-dim_x, self.bbox_center[1]-dim_y], [self.bbox_center[0]+dim_x, self.bbox_center[1]+dim_y] 
            dx_delta = [np.random.randint(self.dx_range[0], self.dx_range[1]), -np.random.randint(self.dx_range[0], self.dx_range[1])]
            self.dx_delta = dx_delta[np.random.randint(2)]
            dy_delta = [np.random.randint(self.dy_range[0], self.dy_range[1]), -np.random.randint(self.dy_range[0], self.dy_range[1])]
            self.dy_delta = dy_delta[np.random.randint(2)]
        else:
            self.px_m = np.random.uniform(self.part_x)
            self.py_m = np.random.uniform(self.part_y)
            dim_x = abs(int(self.dx_orig*self.px_m))
            dim_y = abs(int(self.dy_orig*self.py_m))
            scaled_bbox = [self.bbox_center[0]-dim_x, self.bbox_center[1]-dim_y], [self.bbox_center[0]+dim_x, self.bbox_center[1]+dim_y]
            dx_delta = [np.random.randint(self.max_dx), -np.random.randint(self.max_dx)]
            self.dx_delta = dx_delta[np.random.randint(2)]
            dy_delta = [np.random.randint(self.max_dy), -np.random.randint(self.max_dy)]
            self.dy_delta = dy_delta[np.random.randint(2)]
        scaled_bbox[0][0] += self.dx_delta
        scaled_bbox[0][1] += self.dy_delta
        scaled_bbox[1][0] += self.dx_delta
        scaled_bbox[1][1] += self.dy_delta
        if(scaled_bbox[0][0]<0): #left
            pad0 = abs(scaled_bbox[0][0])
        else:
            pad0 = 0
        if(scaled_bbox[0][1]<0): # top
            pad1 = abs(scaled_bbox[0][1])
        else:
            pad1 = 0

        if(scaled_bbox[1][0]>W): # right
            pad2 = abs(scaled_bbox[1][0]-W)
        else:
            pad2 = 0
        if(scaled_bbox[1][1]>H): # bot
            pad3 = abs(scaled_bbox[1][1]-H)
        else:
            pad3 = 0
        self.padding = list(map(int, [pad0,pad1,pad2,pad3]))    

        if(sum(self.padding)>0):
            self.bbox[0][0]+=pad0
            self.bbox[0][1]+=pad1
            self.bbox[1][0]+=pad0
            self.bbox[1][1]+=pad1
            self.bbox_center[0]+=pad0
            self.bbox_center[1]+=pad1
            scaled_bbox[0][0]+=pad0
            scaled_bbox[0][1]+=pad1
            scaled_bbox[1][0]+=pad0
            scaled_bbox[1][1]+=pad1
        scaled_bbox[0][0] = int(scaled_bbox[0][0])
        scaled_bbox[0][1] = int(scaled_bbox[0][1])
        scaled_bbox[1][0] = int(scaled_bbox[1][0])
        scaled_bbox[1][1] = int(scaled_bbox[1][1])
        self.scaled_bbox = scaled_bbox
        return self.scaled_bbox, self.padding

            
            
    
    def crop(self,pic, is_mask = False):
        color = [255*is_mask, 255*is_mask, 255*is_mask]
        pic = cv2.copyMakeBorder(pic, self.padding[1], self.padding[3],self.padding[0],self.padding[2], cv2.BORDER_CONSTANT, value = color)
        cropped = pic[self.scaled_bbox[0][1]:self.scaled_bbox[1][1]+1, self.scaled_bbox[0][0]:self.scaled_bbox[1][0]+1]
        if(self.size):
            if(self.keep_ratio):
                cropped = cv2.resize(cropped, (self.size[0], np.ceil(self.size[0]*pic.shape[0]/float(pic.shape[1])).astype('int')))
            else:
                cropped = cv2.resize(cropped, self.size)
        pic = cv2.rectangle(pic, (self.scaled_bbox[0][0], self.scaled_bbox[0][1]), (self.scaled_bbox[1][0], self.scaled_bbox[1][1]), (0,255,0),4)
        return cropped
    

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        self.cropped = np.random.rand()<self.prob
        if('fisheye_bbox' in results.keys()):
            self.bbox = results['fisheye_bbox']
        if(self.cropped):
            self.get_crop_box(img.shape[0:2])
            results['img'] = self.crop(img)
            
            for key in results.get('seg_fields', []):
                results[key] = self.crop(results[key], is_mask=True)
        results['part_x'] = self.px_m
        results['part_y'] = self.py_m
        results['dx'] = self.dx_delta
        results['dy'] = self.dy_delta
        results['cropped'] = self.cropped
        results['resized'] = self.size != None
        results['keep_ratio'] = self.keep_ratio
        results['fisheye_bbox'] = self.bbox
         
        
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'

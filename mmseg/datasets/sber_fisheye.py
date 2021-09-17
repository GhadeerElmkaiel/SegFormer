from .builder import DATASETS
from .custom import CustomDataset
from ..core import eval_metrics
from functools import reduce
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
import os
import mmcv
import os.path as osp
from mmcv.image import imresize

@DATASETS.register_module()
class SberbankDatasetFisheye(CustomDataset):
    """Sber Fisheye.
    """
    CLASSES = (
        'Mirror', 'Glass', 'FUO', 'OOS', 'Floor', 'Background', 'Void')
    PALETTE = [[102, 255, 102], [51, 221, 255], [245, 147, 49], [184, 61, 245], 
        [250, 50, 83], [0, 0, 0],[255,255,255]]

    def __init__(self, **kwargs):
        super(SberbankDatasetFisheye, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        for t in self.pipeline.transforms:
                if 'DistortPinholeToFisheye' in str(t):
                    self.distort_transform = t
                    break
        else:
            self.distort_transform = None

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i]
        for class_data in class_table_data[1:]:
            for metric_data, metric_name in zip(class_data[1:], class_table_data[0][1:]):
                eval_results[f"{class_data[0]}_{metric_name}"] = metric_data
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            results = dict(img_info=img_info, ann_info=img_info['ann'])
            self.pre_pipeline(results)
            seg_map = osp.join(self.ann_dir, results['ann_info']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            results['gt_semantic_seg'] = gt_seg_map
            results['seg_fields'] = ['gt_semantic_seg']
            results['scale'] = 1.0
            results['img_shape'] = gt_seg_map.shape
            if self.distort_transform is not None:
                orig_shape = gt_seg_map.shape
                results = self.distort_transform(results)
                results['gt_semantic_seg'] = imresize(results['gt_semantic_seg'], orig_shape[::-1])
            gt_seg_maps.append(results['gt_semantic_seg'])
        return gt_seg_maps

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        return self.prepare_train_img(idx)
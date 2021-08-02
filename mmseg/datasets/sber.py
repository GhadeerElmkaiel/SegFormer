from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SberbankDataset(CustomDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'Glass', 'Mirror', 'Other optical surface', 'Floor', 'Floor under obstacle', 'Background')

    PALETTE = [[51,221,255], [102,255,102], [184,61,245], [250,50,83],
            [245,147,49], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(SberbankDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

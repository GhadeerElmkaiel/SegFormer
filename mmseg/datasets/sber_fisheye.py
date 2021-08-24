from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SberbankDatasetFisheye(CustomDataset):
    """Sber Fisheye.
    """
    CLASSES = (
        'Glass', 'Mirror', 'Other optical surface', 'Floor', 'Floor under obstacle', 'Background', 'Void')

    PALETTE = [[51,221,255], [102,255,102], [184,61,245], [250,50,83],
            [245,147,49], [0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(SberbankDatasetFisheye, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

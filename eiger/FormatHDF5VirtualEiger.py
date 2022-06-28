from __future__ import absolute_import, division, print_function

from cctbx.eltbx import attenuation_coefficient
from dxtbx.model import ParallaxCorrectedPxMmStrategy
from dxtbx.model import BeamFactory
from dials.array_family import flex
import h5py
import numpy as np
from dxtbx.format.FormatHDF5Eiger2SSRL import FormatHDF5Eiger2SSRL

from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.format.FormatStill import FormatStill

IMG_PATH = "virtual_eiger_ssrl_data"

class FormatHDF5VirtualEiger(FormatHDF5Eiger2SSRL, FormatHDF5, FormatStill):
    """
    """

    def _start(self):
        self.handle = h5py.File(self._image_file, "r")
        self.geom_define()
        self.virtual_imgs = self.handle[IMG_PATH]


    @staticmethod
    def understand(image_file):
        understood = False
        try:
            h = h5py.File(image_file, "r")
            if IMG_PATH in list(h.keys()):
                understood = True
        except:
            pass
        
        return understood

    def get_num_images(self):
        return self.virtual_imgs.shape[0] 

    def get_raw_data(self, idx):
        img = self.virtual_imgs[idx]
        if not img.dtype==np.float64:
            img = img.astype(np.float64)
        return (flex.double(img),)


if __name__=="__main__":
    import sys
    for fname in sys.argv[1:]:
        fmt = FormatHDF5VirtualEiger(fname)
        print("%s has %d image" % (fname, fmt.get_num_images()))

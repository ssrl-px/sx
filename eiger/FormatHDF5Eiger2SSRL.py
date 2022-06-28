from __future__ import absolute_import, division, print_function

from cctbx.eltbx import attenuation_coefficient
from dxtbx.model import ParallaxCorrectedPxMmStrategy
from dxtbx.model import BeamFactory
from dials.array_family import flex
import h5py
import numpy as np
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.format.FormatStill import FormatStill
from dxtbx import IncorrectFormatError
from dxtbx.format.FormatPilatusHelpers import determine_eiger_mask


class FormatHDF5Eiger2SSRL(FormatHDF5, FormatStill):
    """
    """
    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def _start(self):
        self.handle = h5py.File(self._image_file, "r")
        self.geom_define()
        
        shots = self.handle['entry/data']
        self.good_shots = []
        nshots = 0
        for key in shots:
            try:
                dset=shots[key]
                
                nd = dset.shape[0]
                shot_key = ['entry/data/' + key]*nd
                shot_inds = list(range(nshots, nd, 1))
                self.good_shots += list(zip(shot_key, shot_inds ))
            except KeyError:
                pass


    @staticmethod
    def understand(image_file):
        # TODO cleanup understand. Make this a dxtbx Nexus? 
        understood = False
        try:
            h = h5py.File(image_file, "r")
            if 'virtual_eiger_ssrl_data' in list(h.keys()):
                return False
            shots = h['entry/data']
            n = 0
            for key in shots:
                try:
                    _=shots[key]
                    n += 1
                except KeyError:
                    bad_shots = []
            if n > 0:
                understood = True 
        except (ImportError, KeyError, OSError):
            pass
        
        return understood

    def geom_define(self):
        self._beam = self._beam()
        self._detector = self._detector()

    def get_beam(self, index=0):
        return self._beam

    def get_detector(self, index=0):
        return self._detector

    def get_num_images(self):
        return len(self.good_shots)

    def get_raw_data(self, idx):
        key, dset_idx = self.good_shots[idx]
        img = self.handle[key][dset_idx]
        if not img.dtype==np.float64:
            img = img.astype(np.float64)
        return (flex.double(img),)

    def _beam(self):
        wavelength = float(self.handle['entry/instrument/beam/incident_wavelength'][()])
        return BeamFactory.simple(wavelength)

    def _detector(self):
        """ Create an Eiger detector profile (taken from FormatCBFMiniEiger) """
        configuration = self.handle["entry/instrument/detector"]

        distance = float(configuration["detector_distance"][()])
        wavelength = float(self.handle['entry/instrument/beam/incident_wavelength'][()])
        beam_x = float(configuration["beam_center_x"][()])
        beam_y = float(configuration["beam_center_y"][()])

        pixel_x = float(configuration["x_pixel_size"][()])
        pixel_y = float(configuration["y_pixel_size"][()])

        material = configuration["sensor_material"][()].decode()
        thickness = float(configuration["sensor_thickness"][()]) * 1000

        nx = int(configuration["detectorSpecific/x_pixels_in_detector"][()])
        ny = int(configuration["detectorSpecific/y_pixels_in_detector"][()])

        try:
            overload = int(configuration["count_rate_correction_count_cutoff"][()])
        except KeyError:
            # hard-code if missing from Eiger stream header
            overload = 4001400
        underload = -1

        try:
            identifier = configuration["description"][()].decode()
        except KeyError:
            identifier = "Unknown Eiger"

        table = attenuation_coefficient.get_table(material)
        mu = table.mu_at_angstrom(wavelength) / 10.0
        t0 = thickness

        detector = self._detector_factory.simple(
            sensor="PAD",
            distance=distance * 1000.0,
            beam_centre=(beam_x * pixel_x * 1000.0, beam_y * pixel_y * 1000.0),
            fast_direction="+x",
            slow_direction="-y",
            pixel_size=(1000 * pixel_x, 1000 * pixel_y),
            image_size=(nx, ny),
            trusted_range=(underload, overload),
            mask=[],
            px_mm=ParallaxCorrectedPxMmStrategy(mu, t0),
            mu=mu,
        )

        for f0, f1, s0, s1 in determine_eiger_mask(detector):
            detector[0].add_mask(f0 - 1, s0 - 1, f1, s1)

        for panel in detector:
            panel.set_thickness(thickness)
            panel.set_material(material)
            panel.set_identifier(identifier)
            panel.set_mu(mu)

        return detector


if __name__=="__main__":
    import sys
    for fname in sys.argv[1:]:
        fmt = FormatHDF5Eiger2SSRL(fname)
        print("%s has %d image" % (fname, fmt.get_num_images()))

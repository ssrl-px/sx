
import os
import h5py

import numpy as np
from argparse import ArgumentParser

# <><><><><><><><><><><><>

parser = ArgumentParser()
parser.add_argument("scoreFile", type=str, help="path to Jinhu's .txt file containing scores and image ids")
parser.add_argument("outFile", type=str, help="name of the output image file")

parser.add_argument("--minScore", type=int, help="Minimum allowed score for an image to go to the virtual hdf5 file (default=1)", default=1)
args = parser.parse_args()

MASTER_ENTRY = "/entry/data"

# <><><><><><><><><><><><>

det="entry/instrument/detector"
MODEL_KEYS = ['entry/instrument/beam/incident_wavelength',
        det+"/detector_distance",
        det+"/beam_center_x",
        det+"/beam_center_y",
        det+"/x_pixel_size",
        det+"/y_pixel_size",
        det+"/sensor_material",
        det+"/sensor_thickness",
        det+"/detectorSpecific/x_pixels_in_detector",
        det+"/detectorSpecific/y_pixels_in_detector"]


def get_frames_per_h5(master_file):
    master_dir = os.path.dirname(master_file)
    h = h5py.File(master_file, "r")
    img_grp = h['entry/data']
    external_links = list(img_grp.keys())
    linked_files = []
    for l in external_links:
        link = img_grp.get(l, getlink=True)
        link_filename = os.path.join(master_dir, link.filename)
        if os.path.exists(link_filename):
            file_num = int(link_filename.split("_")[-1].split(".")[0])
            linked_files.append((file_num, link_filename, link.path))

    if not linked_files:
        raise OSError("no linked files found in master file %s" % master_file)
    linked_files = sorted(linked_files)
    _,first_file, dset_path = linked_files[0]
    first_h = h5py.File(first_file, "r")
    nframes = first_h[dset_path].shape[0]

    nzeros = len(external_links[0].split("_")[1])
    
    return nframes, nzeros



def get_dset_sample(master_file):
    master_dir = os.path.dirname(master_file)
    h = h5py.File(master_file, "r")
    img_grp = h[MASTER_ENTRY]
    external_links = list(img_grp.keys())
    for l in external_links:
        try:
            dset = img_grp[l]
            break
        except KeyError:
            dset = None
            pass
    if dset is None:
        raise OSError("hdf5 master file %s has no readable external links" % master_file)
    return dset
    

def main(score_file, min_score, out_name):

    lines = open(score_file, "r").readlines()

    header = lines[:2]
    data_dir = header[0].split("{")[1].split()[4]


    entries = lines[2:]
        

    entries = np.array([l.strip().split() for l in entries])
    scores = entries[:,1].astype(int)  # TODO: update with real Jinhu file
    # choose the good ones
    sel = scores >= min_score

    scores = scores[sel]
    # optionally sort by score ? 
    #order = np.argsort(scores)[::-1]
    #fname_info = entries[sel, 0][order]

    fname_info = entries[sel, 0]

    all_master_fnames = [  os.path.join(data_dir , l.split("~")[0]) for l in fname_info]

    all_dset_inds = [int(l.split("~")[1]) for l in fname_info]

    unique_masters = set(all_master_fnames)
    print("getting frames per master file (%d master files)" % len(unique_masters))

    frame_per_master = { name: get_frames_per_h5(name) for name in unique_masters }

    num_imgs = len(all_dset_inds)

    open_masters = {}
    for f in unique_masters:
        open_masters[f] = h5py.File(f, 'r')[MASTER_ENTRY]


    test_dset = get_dset_sample(all_master_fnames[0])
    sdim = test_dset.shape[1]
    fdim = test_dset.shape[2]
    dtype = test_dset.dtype

    Vlayout = h5py.VirtualLayout(shape=(num_imgs, sdim, fdim), dtype=dtype)


    sources = {}

    for i_img, (master_name, dset_idx) in enumerate(zip(all_master_fnames, all_dset_inds)):
        
        nframes, nzeros = frame_per_master[master_name]

        master_file_idx = int(dset_idx / nframes)
        master_file_offset = dset_idx % nframes

        data_grp = open_masters[master_name]

        link_name = "data_%s"
        link_name %= str(master_file_idx+1).zfill(nzeros)
        link = data_grp.get(link_name, getlink=True)
        assert link is not None, "A link here should never be None, something is wrong!"
        print(os.path.basename(master_name), master_file_idx, master_file_offset, link.filename, link.path)
        source_path = os.path.join(MASTER_ENTRY,link_name)
        source_key= master_name, source_path

        if source_key not in sources:
            source_shape = open_masters[master_name][link_name].shape
            assert master_file_offset < source_shape[0], "offset is supposed to be within dset bounds! (offset=%d, source_sh0=%d)" % (master_file_offset, source_shape[0])
            Vsource = h5py.VirtualSource(master_name, source_path, shape=source_shape)
            sources[source_key] = Vsource

        Vlayout[i_img] = sources[source_key][master_file_offset] 


    with h5py.File(out_name, "w") as writer:

        # copy over the detector and beam information directly
        master_h = h5py.File(all_master_fnames[0], 'r')
        for k in MODEL_KEYS:
            try:
                model_val = master_h[k][()]
                writer.create_dataset(k, data=model_val)
                print(k,model_val) 
            except KeyError:
                # NOTE: this should not happen:
                raise KeyError("key %s not in masterfile %s. Something is wrong!" % (k, all_master_fnames[0]))
        
        writer.create_virtual_dataset("virtual_eiger_ssrl_data", Vlayout)


    # make a copy; necessary for input to IOTA
    with h5py.File(out_name, "r+") as writer:
        writer["entry/data/data_000001"] = writer["virtual_eiger_ssrl_data"]
        

    print("Wrote virtual hdf5 file %s with %d shots!" % (out_name, len(scores) ))

if __name__=="__main__":
    main(args.scoreFile, args.minScore, args.outFile)

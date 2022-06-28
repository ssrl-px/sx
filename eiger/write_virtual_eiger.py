
import os
import h5py

import numpy as np
import glob

# <><><><><><><><><><><><>

MASTER_ENTRY = "/entry/data"

_det="entry/instrument/detector"
MODEL_KEYS = ['entry/instrument/beam/incident_wavelength',
        _det+"/detector_distance",
        _det+"/beam_center_x",
        _det+"/beam_center_y",
        _det+"/x_pixel_size",
        _det+"/y_pixel_size",
        _det+"/sensor_material",
        _det+"/sensor_thickness",
        _det+"/detectorSpecific/x_pixels_in_detector",
        _det+"/detectorSpecific/y_pixels_in_detector"]

SCORE_POS = 3
NSPOTS_POS = 1

# <><><><><><><><><><><><>

def get_frames_per_h5(master_file, verbose=False):
    master_dir = os.path.dirname(master_file)
    h = h5py.File(master_file, "r")
    img_grp = h['entry/data']
    external_links = list(img_grp.keys())
    if verbose:
        print(external_links)
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


def dataDir_to_virtual(dat_dir, out_name):
    """
    data_dir, path to a folder containing master files
    out_name, output hdf5 name; will be FormatHDF5VirtualEiger format
    """
    fnames = glob.glob("%s/*master*" % dat_dir)
    all_entries = []
    for i_f, f in enumerate(fnames):
        h = h5py.File(f, 'r')
        G = h['entry/data']
        nfiles =G.keys()
        nfiles = sorted(nfiles, key=lambda x: int(x.split("_")[-1]))
        offset = 0
        entries =[]
        for k in nfiles:
            link = G.get(k, getlink=True)
            
            filepath = os.path.join(os.path.dirname(f), link.filename)
            if not os.path.exists(filepath):
                continue
            dset = h5py.File(filepath, 'r')[link.path]
            print(os.path.basename(f), os.path.basename(filepath), dset.shape[0])
            n = dset.shape[0]
            global_inds = np.arange(offset, offset+n,1)
            entries += list(zip([os.path.basename(f)]*n, global_inds))
            offset += n
        all_entries+= entries

    fname_info = []
    for entry in all_entries:
        s = "%s~%d" % entry
        print(s)
        fname_info.append(s)

    main(fname_info, dat_dir, out_name)
    

def scoreFile_to_virtual(score_file, min_score, out_name, min_nspots=10):
    """
    score_file: path to Jinhus score .txt file
    min_score: minimum score allowed 
    out_name: output hdf5 name, will be FormatHDF5VirtualEiger format
    """

    lines = open(score_file, "r").readlines()

    header = lines[:2]
    data_dir = header[0].split("{")[1].split()[4]


    entries = lines[2:]
        

    entries = np.array([l.strip().split() for l in entries])
    scores = entries[:,SCORE_POS].astype(int)  # TODO: update with real Jinhu file
    nspots = entries[:,NSPOTS_POS].astype(int)  # TODO: update with real Jinhu file
    # choose the good ones
    sel = (scores >= min_score ) * ( nspots >= min_nspots)
    if not np.any(sel):
        print("No entries selected for virtualization, lower minScore and/or minNpots")
        return

    scores = scores[sel]
    nspots = nspots[sel]
    # optionally sort by score ? 
    #order = np.argsort(scores)[::-1]
    #scores = scores[order]
    #fname_info = entries[sel, 0][order]

    fname_info = entries[sel, 0]

    main(fname_info, data_dir, out_name, False)


def main(fname_info, data_dir, out_name, zero_based_indexing=True):
    """
    fname_info: list of "master_file_basename.h5~dset_idx" where dset_idx is an integer specifying image name within the master file
    data_dir: path where master files are stored
    out_name: output hdf5 name
    """
    if len(fname_info) ==0 :
        print("Nothing to process! fname_info has 0 length")
        return

    all_master_fnames = [  os.path.join(data_dir , l.split("~")[0]) for l in fname_info]

    offset = 0
    if not zero_based_indexing:
        offset = 1
    all_dset_inds = [int(l.split("~")[1])-offset for l in fname_info]

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
        print(i_img, os.path.basename(master_name), master_file_idx, master_file_offset, link.filename, link.path)

        source_path = os.path.join(MASTER_ENTRY,link_name)
        source_key= master_name, source_path

        if source_key not in sources:
            source_shape = open_masters[master_name][link_name].shape
            assert master_file_offset < source_shape[0], "offset is supposed to be within dset bounds! (offset=%d, source_sh0=%d)" % (master_file_offset, source_shape[0])
            Vsource = h5py.VirtualSource(master_name, source_path, shape=source_shape)
            sources[source_key] = Vsource

        try:
            SOURCE = sources[source_key][master_file_offset]
        except:
            from IPython import embed;embed()

        Vlayout[i_img] = SOURCE #$sources[source_key][master_file_offset] 


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
        

    print("Wrote virtual hdf5 file %s with %d shots!" % (out_name, len(fname_info) ))


if __name__=="__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input", type=str, help="either a datadir with master files -OR- a path to Jinhu's .txt file containing scores and image ids")
    parser.add_argument("outFile", type=str, help="name of the output image file")

    parser.add_argument("--minScore", type=int, help="Minimum allowed score for an image to go to the virtual hdf5 file (default=1)", default=1)
    parser.add_argument("--minNspots", type=int, help="Minimum number of spots for an image to go to the virtual hdf5 file (default=10)", default=10)
    args = parser.parse_args()
    args = parser.parse_args()

    if os.path.isdir(args.input):
        dataDir_to_virtual(args.input, args.outFile)
    else:
        scoreFile_to_virtual(args.input, args.minScore, args.outFile, args.minNspots)


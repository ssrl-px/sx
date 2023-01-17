
import time
import numpy as np
import h5py
import pylab as plt
import torch

from resnet import net
from sx.diffraction_ai import maxbin


MODEL_A = "/data/blstaff/xtal/mwilson_data/diff_AI/nety_ep40.nn"

def load_model(state_name):
    model = net.RESNet50()
    state = torch.load(state_name)
    model.load_state_dict(state)
    model = model.to("cpu")
    model = model.eval()
    return model
    

#MASK = np.load("/data/blstaff/xtal/mwilson_data/mask_mwils.npy")

def raw_img_to_tens(raw_img, MASK):
    img = maxbin.get_quadA(maxbin.img2int(raw_img*MASK))
    img = img.astype(np.float32)[:512,:512]
    img_t = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img_t

def raw_img_to_tens_pil(raw_img, MASK, xy=None):
    ysl, xsl = maxbin.get_slice_pil(xy)
    # or else pad img if shape is not 1024x1024
    img = maxbin.img2int_pil(raw_img[ysl, xsl]*MASK[ysl,xsl])    
    img = maxbin.get_quadA_pil(img).astype(np.float32)
    img_t = torch.tensor(img).view((1,1,512,512)).to("cpu")
    return img_t
 

#fit_rad = model2(img_t).item()

#ax = plt.gca()
#ax.imshow(np.zeros((512,512)), vmin=0, vmax=10)
#
#for i in range(imgs.shape[0]):
#    t = time.time()
#    raw_img = raw_imgs[i]
#    #from IPython import embed;embed()
#    t2= time.time()
#    img = maxbin.get_quadA(maxbin.img2int(raw_img*MASK)).astype(np.float32)[:512,:512]
#    t2= time.time()-t2
#    #from IPython import embed;embed()
#    #img = imgs[i,0]
#    #img [ img > 120] = 0
#    #from IPython import embed;embed()
#    t3= time.time()
#    img_t = torch.tensor(img).view((1,1,512,512)).to("cpu")
#    t3= time.time()-t3
#    #a = a.view(torch.Size((1,1,512,512)))
#    #a = a.to("cpu")
#
#    
#    #fit_rad = model(a).item()
#    t4 = time.time()
#    fit_rad = model2(img_t).item()
#    t4 = time.time()-t4
#    t= time.time()-t
#    print(i, "fit radius=%.3f (took %.4f sec), maxbin=%.3f, tens=%.3f, eval=%.3f" % (fit_rad,t,t2,t3,t4) )
#    continue
#    ax.images[0].set_data(img)
#    try:
#        ax.patches.pop()
#    except IndexError:
#        pass
#    
#    #plt.cla();plt.imshow(img, vmax=10)
#    plt.title("Shot %d, AI-predicted radius=%.1f" % (i+1, fit_rad))
#    #plt.gca().add_patch(plt.Circle(xy=(0,0), radius=fit_rad, fc='none', ec='r', ls='--', label="predicted radius"))
#    ax.add_patch(plt.Circle(xy=(0,0), radius=fit_rad, fc='none', ec='r', ls='--', label="predicted radius"))
#    plt.legend(loc=4)
#    plt.draw();plt.pause(0.7)
#    #plt.savefig("test_resonet_integ2_trial.23_fits/shot%04d.png" % i, dpi=150)
#

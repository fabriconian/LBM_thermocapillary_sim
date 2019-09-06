import numpy as np

def convolve(x,k):

    h_filter = k.shape[0]
    w_filter = k.shape[1]
    c = k.shape[2]
    c_out = k.shape[3]
    h_out = x.shape[1] - h_filter + 1
    w_out = x.shape[2] - w_filter + 1

    out = np.empty((x.shape[0], h_out, w_out, c_out))
    temp = 0.0
    pad_h =int( h_filter / 2)
    pad_w = int(w_filter / 2)
    out = np.empty((x.shape[0], h_out, w_out, c_out))
    temp = 0.0
    if pad_w >0:
        for i_x in range(h_out):
            for j_x in range(w_out):
                for c_f in range(c_out):
                    for c_x in range(c):
                        for i_f in range(-pad_h,pad_h+1):
                            for j_f in range(-pad_w,pad_w+1):
                                temp+= x[0,i_x+i_f+pad_h,j_x+j_f+pad_w, c_x]*k[i_f + pad_h, j_f + pad_w,c_x,c_f]

                    out[0,i_x, j_x, c_f] = temp
                    temp = 0.0
    else:
        for i_x in range(h_out):
            for j_x in range(w_out):
                for c_f in range(c_out):
                    for c_x in range(c):
                        temp+= x[0,i_x,j_x, c_x]*k[0, 0,c_x,c_f]

                    out[0,i_x, j_x, c_f] = temp
                    temp = 0.0

    return out

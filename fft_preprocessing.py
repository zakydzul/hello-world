def fft_preprocessing(data,f):
    for ar in data:
        L = ar.shape[1]
        l=[]
        for i in range(ar.shape[0]):
            Y1 = np.fft.fftn(ar[i])
            P1 = 2*np.abs(Y1/L)
            x_fft = np.square(P1[0:L//2])
            # plt.plot(f,x_fft)
        
            p = interpolate.interp1d(f,x_fft.flatten())
        
            x_new = np.arange(0,f[-1],1)
            y_new = p(x_new)
            l.append(y_new)
            
        l = np.array(l)
        l = minmax_scale(l,axis=1)
        fft_data.append(l)
    return fft_data, len(x_new)

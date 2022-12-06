function [seis,twave]=arrivaltoseisresolution(A_t,n,nt,dt,type)


if type == 1
    ntpow2 = 2^(floor(log2(nt))+1);
    nfreq  = 2*ntpow2;
    df     = 1./(nfreq*dt);
    dom    = 2.*pi*df;


    tend = (nfreq-1)*dt;
    freq_source = 10;
    
    % create a time vector
    nt=round(tend/dt)+1;
    tmin=-dt*round(nt/2);
    twave=tmin+dt*(0:nt-1)';
    % create the wavelet
    pf=pi^2*freq_source^2;
    wavelet=(1-2.*pf*twave.^2).*exp(-pf*twave.^2);
    
    % normalize
    % generate a refenence sinusoid at the dominant frequency
    wavelet=wavenorm(wavelet,twave,2);
    
    cwav = complex(zeros(1,nfreq),zeros(1,nfreq));
    for it=1:ntpow2
        cwav(it) = complex(wavelet(it),0.);
    end 
    for it=ntpow2+1:2*ntpow2
        cwav(it) = complex(wavelet(it),0.0);
    end

    fftwav = fft(cwav);

    cf     = complex(zeros(1,nfreq),zeros(1,nfreq));
    seis   = zeros(nfreq,n);

    for ix=1:n

        for ifreq=1:nfreq/2+1
            om        = (ifreq-1)*dom;
            cf(ifreq) = fftwav(ifreq)*exp(1i*om*A_t(ix,1));
        end 
    %   take the complex conjugate for the negative frequencies
            for ifreq=nfreq/2+2:nfreq
                ifreq1    = nfreq-ifreq+2;
                rr        = real(cf(ifreq1));
                ri        = imag(cf(ifreq1));
                cf(ifreq) = complex(rr,-ri);
            end

        seis(:,ix) = real(ifft(cf')) ; % 
    end%ix-loop
elseif type == 2 
    ntpow2 = 2^(floor(log2(nt))+1);
    nfreq  = 2*ntpow2;
    df     = 1./(nfreq*dt);
    dom    = 2.*pi*df;


    tlength = (nfreq-1)*dt;
    fdom = 10;

    % create a time vector
      nt=round(tlength/dt)+1;
      tmin=-dt*round(nt/2);
      twave=tmin+dt*(0:nt-1)';
    % create the wavelet
      wavelet=sinc(2*pi*fdom*twave); %(1-2.*pf*tw.^2).*exp(-pf*tw.^2);

    % normalize
    % generate a refenence sinusoid at the dominant frequency
    wavelet=wavenorm(wavelet,twave,2);


    cwav = complex(zeros(1,nfreq),zeros(1,nfreq));
    for it=1:ntpow2
        cwav(it) = complex(wavelet(it),0.);
    end 
    for it=ntpow2+1:2*ntpow2
        cwav(it) = complex(wavelet(it),0.0);
    end

    fftwav = fft(cwav);

    cf     = complex(zeros(1,nfreq),zeros(1,nfreq));
    seis   = zeros(nfreq,n);

    for ix=1:n

        for ifreq=1:nfreq/2+1
            om        = (ifreq-1)*dom;
            cf(ifreq) = fftwav(ifreq)*exp(1i*om*A_t(ix,1));
        end 
    %   take the complex conjugate for the negative frequencies
            for ifreq=nfreq/2+2:nfreq
                ifreq1    = nfreq-ifreq+2;
                rr        = real(cf(ifreq1));
                ri        = imag(cf(ifreq1));
                cf(ifreq) = complex(rr,-ri);
            end

        seis(:,ix) = 0.75*real(ifft(cf')) ; % 
    end%ix-loop   
end  
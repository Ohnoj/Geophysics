% addpath(genpath('C:\Users\TU Delf SID\Documents\MATLAB\ScienceCentre_HonoursProject\crewes'))
close all
clear all
clc

dt = 0.0004;
nt = 1000;
rx = 1:2;

type = 1; % Defines Rickert (1) or Sinc (2) wavelet

vidObj=VideoWriter('BothPolarityRickComplete.avi');
vidObj.FrameRate=10;
open(vidObj);
figure()

set(gcf,'Position',[100 100 1820 980],'Color','w')

for it = 0:0.0025:0.2
    [seis1,twave1] = arrivaltoseisresolution(ones(numel(rx))*0.8 + it,numel(rx),nt,dt,type);
    [seis2,twave2] = arrivaltoseisresolution(ones(numel(rx))*1,numel(rx),nt,dt,type);
        
    twave1 = twave1 - min(twave1);
    twave2 = twave2 - min(twave2);

    seis = [seis1(:,1)+seis2(:,1) seis1(:,2)-seis2(:,2)];
%     figure()
    AX = subplot(1,2,1);
    plotseis(seis,0:0.0004:0.0004*2047,rx,0,[1.5 0.04],1,1,'r');
    set(gca,'XTick',[1 2],'XTickLabel',{'Same Polarity','Opposite Polarity'})
    ylabel('Time [s]')
    line([1 1.2],[0.39+it 0.39+it])
    line([2 2.2],[0.39+it 0.39+it])
    line([1 1.2],[0.59 0.59])
    line([1.8 2],[0.59 0.59])
   
    
    subplot(1,2,2)
    [ax,h1,h2] = plotyy([0 0],[0 0],[0 0],[0 0]);
    hold(ax(1),'on');
    set(ax(1),'Box','off')
    set(ax,'YColor','k')
    ylabel(ax(1),'Apparent TWT-Thickness [s]')
    xlabel('True TWT-Thickness [s]')
    ylabel(ax(2),'Normalised Amplitude')
    hold(ax(2),'on');
    ylim(ax(1),[0 0.02]) % sinc 0.04, rick 0.02
    xlim(ax(1),[0 0.2])
    xlim(ax(2),[0 0.2])
    ylim(ax(2),[0 0.2])
    set(ax(2),'YTick',[0 0.1925/3 0.1925/2 0.1925*2/3 0.1925],'YTickLabel',[0 0.5 1 1.5 2])
    set(ax(1),'YTick',[0 0.002 0.004 0.006 0.008 0.01 0.012 0.014 0.016 0.018 0.02],'YTickLabel',[0 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.016 0.18 0.2])
    plot(ax(2),[0 0.2],[0 0.2],'k-.')
    
    OP = scatter(ax(1),1-0.8-it,abs(max(seis(:,2))),'rp','fill');
    SP = scatter(ax(1),1-0.8-it,abs(max(seis(:,1))),'bp','fill');
    
    [~,locsPtP] = findpeaks(seis(:,1),'MinPeakHeight',max(seis(:,1))-0.001);
    if numel(locsPtP) > 1
        PtP = abs(twave1(locsPtP(1))-twave1(locsPtP(2))); 
    else
        PtP = 0;
    end
    PtT = abs(twave1(seis(:,2) == max(seis(:,2)))-twave1(seis(:,2) == min(seis(:,2))));    
    
    OP2 = scatter(ax(1),1-0.8-it,PtT(1)/(0.2)*0.02,'r.');
    SP2 = scatter(ax(1),1-0.8-it,(PtP(1)/(0.2)*0.02),'b.');
    legend([SP OP SP2 OP2],{'Same Polarity Amplitude','Opposite Polarity Amplitude','Same Polarity Apparent Thickness','Opposite Polarity Apparent Thickness'},'Location','SouthEast');
    
    
    
    if it == 64*0.0025
        tl = 0;
        while tl < 10
        p=getframe(gcf);
        writeVideo(vidObj,p);
        tl = tl + 1;
        end
    end
    if it == 67*0.0025
        tl = 0;
        while tl < 10
        p=getframe(gcf);
        writeVideo(vidObj,p);
        tl = tl + 1;
        end
    end
    
    p=getframe(gcf);
    writeVideo(vidObj,p);
    cla(AX)
end
close all
close(vidObj);
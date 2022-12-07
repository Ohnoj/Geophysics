# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:05:52 2018

@author: Johno van IJsseldijk
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Cursor
plt.rcParams['keymap.xscale'] = 'None'
plt.rcParams['keymap.yscale'] = 'None'

def onkey(event):
    if event.key == 'r':
        reset()

def onclick(event):
    if event.button == 1:
        cursor.set_active(False)
    elif event.button == 2:
        cursor.set_active(True)
    fig.canvas.draw()
    
def update(val):
    upm = sm.val
    upR0 = sR0.val
    uptau = stau.val
    upc = sc.val
    ups = upR0 * (1 - upm*(1-(1)/((1+(1j*omega*2*np.pi*uptau)**upc))))
    l.set_ydata( abs(ups) )
    k.set_ydata( -np.angle(ups)*1000 )
    fig.canvas.draw_idle()

def reset():
    sm.reset()
    sR0.reset()
    stau.reset()
    sc.reset()


fig, ax1 = plt.subplots()
plt.subplots_adjust(top=0.95,bottom=0.25,right=.85,left=.2)
plt.connect('key_press_event',onkey)
plt.connect('button_press_event',onclick)
plt.title('Amplitude and phase curves of the Cole-Cole relaxation model',fontsize=16,fontweight='bold')
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
cursor = Cursor(ax1, useblit=True, color='green', linewidth=2,  alpha=0.7)
cursor.set_active(False)
plt.grid()

omega = np.logspace(-2,5,1000)
m0 = .5
R00 = 1.
tau0 = 1e-2
c0 = .25
s = R00 * (1 - m0*(1-(1)/((1+(1j*omega*2*np.pi*tau0)**c0))))
l, = ax1.loglog(omega, abs(s), lw=2, color='red',label='Amplitude')
ax1.set_ylabel('Amplitude [$\Omega$]')
ax1.set_xlabel('Frequency [Hz]')
#ax1.set_xlim([1e-2,1e8])
ax1.set_ylim([1e-3, 10])
ax2 = ax1.twinx()
ax2.set_ylabel('Phase [mRad]')
ax2.set_ylim([1, 1e3])

k, = ax2.loglog(omega, -np.angle(s)*1000, 'b--', lw=2,label='Phase')
plt.legend(handles = [l,k],loc='best')

axcolor = 'lightgoldenrodyellow'

#Sliders
m = plt.axes([0.2, 0.16, 0.65, 0.03], facecolor=axcolor)
R0 = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor=axcolor)
#R0.set_xscale('log')
#R0.tick_params(axis='x',which='both',top='off',bottom='off',labelbottom='off')
tau = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor=axcolor)
tau.set_xscale('log')
tau.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',
    labelbottom='off')
c = plt.axes([0.2, 0.11, 0.65, 0.03], facecolor=axcolor)

sm = Slider(m, 'm', 0.1, 0.9, valinit=m0)
sR0 = Slider(R0, 'R$_0$', 0.2, 5.0, valinit=R00)
stau = Slider(tau, '$\\tau$', 1e-4, 1e2, valinit=tau0,valfmt='%.1e')
sc = Slider(c, 'c', 0.1, 0.6, valinit=c0)

sm.on_changed(update)
sR0.on_changed(update)
stau.on_changed(update)
sc.on_changed(update)


plt.show()

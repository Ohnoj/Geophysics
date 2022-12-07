# -*- coding: utf-8 -*-
"""
Engineering Geophysics: Task 13 Image Processing
Method: Magnetic vertical gradient data
Location: Mediant Watfa, NW Fayum Oasis, Egypt
Target: Granary from early Ptolemaic period (~300 BC) 
Johno van IJsseldijk - 04/12/17 
MSc. Applied Geophysics
"""

"""
0. Import Libraries & set universal functions
"""

import glob
import tkinter as tk
import sys
import copy
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
Keymaps = ['keymap.all_axes',
           'keymap.back',
           'keymap.forward',
           'keymap.grid',
           'keymap.home',
           'keymap.pan',
           'keymap.save',
           'keymap.xscale',
           'keymap.yscale',
           'keymap.zoom']
for item in Keymaps:
    plt.rcParams[item] = 'None'
    
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Cursor
from scipy.signal import convolve2d




def onclick(event):
    global curon
    if event.button == 1:
        pass
    if event.button == 2 and not curon:
        curon = True
    elif event.button == 2 and curon:
        curon = False      

def onkey(event):
    global figbreak
    x,y = event.xdata,event.ydata
    if event.key == 'q' and x != None:
        figbreak = True
    if event.key == 'd' and x != None: 
        destagger(x,y)
    if event.key == 'e' and x != None:
        edgematch(x,y)
    if event.key == 'z' and x != None:
        zeromeantraverse(x,y)
    if event.key == 'i' and x != None:
        interpolategrid()
    if event.key == 'escape' and x != None:
        print('Resetting')
        reset()
    if event.key == 'l' and x != None:
        low_pass_filter()
    plt.draw()
    
def get_current_grid(x,y):
    if x <= 19.75:
        gridx = 1
    elif x <= 39.75:
        gridx = 2
    else:
        gridx = 3
    if y <= 19.5:
        gridy = 1
    elif y <= 39.5:
        gridy = 2
    else:
        gridy = 3
    grid = (gridx-1)+3*(gridy-1)
    return grid

def concatenate_all(im):
    return np.concatenate((np.concatenate((im[:,:,0],im[:,:,1],im[:,:,2]),axis=1),
                         np.concatenate((im[:,:,3],im[:,:,4],im[:,:,5]),axis=1),
                         np.concatenate((im[:,:,6],im[:,:,7],im[:,:,8]),axis=1)))

def reset():
    global IM_raw
    global im_raw
    global IM_process
    global im_process
    IM_process = IM_raw
    im_process = im_raw
    ax = plt.gca()
    ax.set_title('Original Image')

"""
1. Loading data
""" 

filelist = (glob.glob("DataNew/*.dat"))

order = [1,0,8,5,7,4,6,3,2]
Data = np.zeros([3200,3,len(filelist)],dtype = float);
g = np.zeros([3200,3], dtype = float);
im_raw = np.zeros([40,80,len(filelist)], dtype = float);
fig = plt.figure()


for i in range(0,len(filelist)):
    Data[:,:,i] = np.genfromtxt(filelist[i],
                             dtype=float,
                             delimiter=',')
    G = Data[:,:,i]
    x = G[:,0]
    y = G[:,1]
    d = G[:,2]
    X = x.reshape((40,80))
    Y = y.reshape((40,80))
    D = d.reshape((40,80))
    im_raw[:,:,order[i]] = D


names =['335','333','27','336','334','28','253','251','55'] 
locs = [6,7,8,3,4,5,0,1,2]
for i in range(0,len(filelist)):
    ax = fig.add_subplot(33*10+locs[i]+1)    
    ax.imshow(im_raw[:,:,i],extent=[0, 19.75, 19.5, 0],cmap=plt.get_cmap('Greys'),clim = [-10,10], aspect='equal')
    ax.set_title(names[locs[i]]+'.dat')
    ax.invert_yaxis()
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()

IM_raw = concatenate_all(im_raw)

"""
2. Destaggering the data
"""  

class Get_shift(tk.Tk):
    
    def __init__(self,grid):
        global shift
        grids = [253,251,55,336,334,28,335,333,27]
        tk.Tk.__init__(self)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.entry = tk.Entry()
        self.label1 = tk.Label(text='Destaggering grid {:d}'.format(grids[grid]))
        self.label2 = tk.Label(text='Horizontal shift')
        self.button = tk.Button(text="Destagger", command=self.on_button)
        self.entry.insert(10,shift)
        self.label1.pack()
        self.label2.pack()
        self.entry.pack()
        self.button.pack()
        self.geometry('150x95+920+450')
        self.alive = True
        
    def on_closing(self):
        global shift
        shift = 0
        self.alive = False
        tk.Tk.destroy(self)
        del self
    
    def on_button(self):
        global shift
        try:
            shift = int(self.entry.get())
        except:
            shift = 0
        self.alive = False
        tk.Tk.destroy(self)
        del self
        


def destagger(x,y):
    global im_raw
    global im_process
    global IM_process
    global shift
    ax = plt.gca()
    ax.set_title('Destaggering')
    grid = get_current_grid(x,y)
    try:
        if not root1.alive:
            root1 = Get_shift(grid)
            root1.mainloop()
    except NameError:
        root1 = Get_shift(grid)
        root1.mainloop()
    if shift > 0:
        for i in range(0,39,2):
            if i == 0:
                im_process[i,:,grid] = np.concatenate((im_process[i+1,:shift,grid],im_process[i,:-shift,grid]))
            else:
                im_process[i,:,grid] = np.concatenate((np.mean(np.array((im_process[i-1,:shift,grid],im_process[i+1,:shift,grid])),axis=0),im_process[i,:-shift,grid]))
    elif shift < 0:
        for i in range(0,39,2):
            if i == 0:
                im_process[i,:,grid] = np.concatenate((im_process[i,-shift:,grid],im_process[i+1,shift:,grid]))
            else:
                im_process[i,:,grid] = np.concatenate((im_process[i,-shift:,grid],np.mean(np.array((im_process[i-1,shift:,grid],im_process[i+1,shift:,grid])),axis=0)))
    IM_process = concatenate_all(im_process)

"""
3. Edge matching
""" 

class Checkbar(tk.Frame):
   def __init__(self, parent=None, picks=[], anchor='w'):
      tk.Frame.__init__(self, parent)
      self.vars = []
      for pick in picks:
         var = tk.IntVar()
         chk = tk.Checkbutton(self, text=pick, variable=var)
         chk.pack(anchor=anchor, expand='no')
         self.vars.append(var)
   def state(self):
      return map((lambda var: var.get()), self.vars)

class Edges2Match(tk.Tk):
    
    def __init__(self,grid):
        grids = [253,251,55,336,334,28,335,333,27]
        tk.Tk.__init__(self)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.label1 = tk.Label(text='Edgematching grid: {:d}'.format(grids[grid]))
        self.button = tk.Button(text="Match Edges", command=self.on_button)
        self.E2M1 = Checkbar(self,['Top   ','Right'])
        self.E2M2 = Checkbar(self,['Bottom','Left '])
        self.label1.pack(side ='top',fill = 'x',anchor='n',expand=True)
        self.button.pack(side = 'bottom')
        self.E2M1.pack(side='left',fill= 'x')
        self.E2M2.pack(side='right',fill= 'x')
        self.geometry('150x95+920+450')
        self.alive = True
        
    def on_closing(self):
        global E2M
        self.alive = False
        E2M = [0]*4
        tk.Tk.destroy(self)
        del self
    
    def allstates(self):
        global E2M
        E2M = list(self.E2M1.state()) + list(self.E2M2.state())
        
    def on_button(self):
        self.allstates()
        self.alive = False
        tk.Tk.destroy(self)
        del self

def edgematch(x,y):
    global im_process
    global IM_process
    global E2M
    grid = get_current_grid(x,y)
    limits = [-10,10]
    try:
        if not root.alive():
            root = Edges2Match(grid)
            root.mainloop()
    except NameError:
        root = Edges2Match(grid)
        root.mainloop()
    
    if all([E2M[0],grid!=8,grid!=7,grid!=6]):
        print('matching top edges')
        temp1 = im_process[-3:,:,grid]
        temp2 = im_process[:3,:,grid+3]
        Diff = np.mean(temp1[np.logical_and(temp1>limits[0],temp1<limits[1])])-np.mean(temp2[np.logical_and(temp2>limits[0],temp2<limits[1])])
        im_process[:,:,grid+3] += Diff
    if all([E2M[1],grid!=2,grid!=5,grid!=8]):
        print('matching right edges')
        temp1 = im_process[:,-3:,grid]
        temp2 = im_process[:,:3,grid+1]
        Diff = np.mean(temp1[np.logical_and(temp1>limits[0],temp1<limits[1])])-np.mean(temp2[np.logical_and(temp2>limits[0],temp2<limits[1])])
        im_process[:,:,grid+1] += Diff
    if all([E2M[2],grid!=1,grid!=2,grid!=0]):
        print('matching bottom edges')
        temp1 = im_process[:3,:,grid]
        temp2 = im_process[-3:,:,grid-3]
        Diff = np.mean(temp1[np.logical_and(temp1>limits[0],temp1<limits[1])])-np.mean(temp2[np.logical_and(temp2>limits[0],temp2<limits[1])])
        im_process[:,:,grid-3] += Diff
    if all([E2M[3],grid!=6,grid!=0,grid!=3]):
        print('matching left edges')
        temp1 = im_process[:,3:,grid]
        temp2 = im_process[:,-3:,grid-1]
        Diff = np.mean(temp1[np.logical_and(temp1>limits[0],temp1<limits[1])])-np.mean(temp2[np.logical_and(temp2>limits[0],temp2<limits[1])])
        im_process[:,:,grid-1] += Diff
    ax = plt.gca()
    ax.set_title('Edge Matching')
    IM_process = concatenate_all(im_process)
    

"""
4. Zero Mean Traverse
"""

class ZMT(tk.Tk):
    
    def __init__(self,grid):
        tk.Tk.__init__(self)
        grids = [253,251,55,336,334,28,335,333,27]
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.label1 = tk.Label(text='Would you like to perform\nzero mean traverse on grid {:d}?'.format(grids[grid]))
        self.button1 = tk.Button(text="Yes", command=self.on_button1)
        self.button2 = tk.Button(text="No", command=self.on_button2)
        self.label1.pack(fill = "both",expand=True)
        self.button1.pack(side="left", fill="both", expand=True)
        self.button2.pack(side="right", fill="both", expand=True)
        self.geometry('220x75+920+450')
        self.alive = True
        
    def on_closing(self):
        global ZMTb
        ZMTb = False
        self.alive = False
        tk.Tk.destroy(self)
        del self
    
    def on_button1(self):
        global ZMTb
        ZMTb = True
        self.alive = False
        tk.Tk.destroy(self)
        del self
        
    def on_button2(self):
        global ZMTb
        ZMTb = False
        self.alive = False
        tk.Tk.destroy(self)
        del self

def zeromeantraverse(x,y):
    global ZMTb
    global im_process
    global IM_process
    limits = [-10,10]
    grid = get_current_grid(x,y)
    try:
        if not root4.alive():
            root4 = ZMT(grid)
            root4.mainloop()
    except NameError:
        root4 = ZMT(grid)
        root4.mainloop()
    if ZMTb:
        print('ZMTing grid')
        for i in range(0,40):
            traverse = im_process[i,:,grid]
            im_process[i,:,grid] -= np.mean(traverse[np.logical_and(traverse>limits[0],traverse < limits[1])])        
    IM_process = concatenate_all(im_process)  
    ax = plt.gca()
    ax.set_title('Zero Mean Traverse')

"""
5. Interpolation
"""

class Int_image(tk.Tk):
    
    def __init__(self):
        tk.Tk.__init__(self)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.label1 = tk.Label(text='Do you want to interpolate the image?')
        self.button1 = tk.Button(text="Yes", command=self.on_button1)
        self.button2 = tk.Button(text="No", command=self.on_button2)
        self.label1.pack(fill = "both",expand=True)
        self.button1.pack(side="left", fill="both", expand=True)
        self.button2.pack(side="right", fill="both", expand=True)
        self.geometry('220x75+920+450')
        self.alive = True
        
    def on_closing(self):
        global interpolate_image
        interpolate_image = False
        self.alive = False
        tk.Tk.destroy(self)
        del self
    
    def on_button1(self):
        global interpolate_image
        interpolate_image = True
        self.alive = False
        tk.Tk.destroy(self)
        del self
        
    def on_button2(self):
        global interpolate_image
        interpolate_image = False
        self.alive = False
        tk.Tk.destroy(self)
        del self

def interpolategrid():
    global IM_process
    global interpolate_image
    interpolate_image = False
    try:
        if not root2.alive:
            root2 = Int_image()
            root2.mainloop()
    except NameError:
        root2 = Int_image()
        root2.mainloop()
    if interpolate_image:
        ax = plt.gca()
        ax.set_title('Interpolated Image')
        x = np.arange(0.,60.,.25)
        y = np.arange(0.,60.,.5)
    
        f = interp.interp2d(x, y, IM_process)
        IM_process = f(x,x)

"""
6. Low-pass Filter
"""

class lpf_image(tk.Tk):
    
    def __init__(self):
        tk.Tk.__init__(self)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.label1 = tk.Label(text='Window for low-pass filter:')
        self.entry1 = tk.Entry()
        self.entry2 = tk.Entry()
        self.button1 = tk.Button(text="Filter", command=self.on_button1)
        self.label1.pack(fill = "both",expand=True)
        self.entry1.insert(10,5)
        self.entry2.insert(10,5)
        self.button1.pack(side='bottom',expand=False)
        self.entry1.pack(side="left",fill = 'both',expand=False)
        self.entry2.pack(side="right",fill = 'both',expand=False)

        self.geometry('180x75+920+450')
        self.alive = True
        
    def on_closing(self):
        global applyfilter
        applyfilter = False
        self.alive = False
        tk.Tk.destroy(self)
        del self
    
    def on_button1(self):
        global applyfilter
        global window
        try:
            window[0] = abs(int(self.entry1.get()))
            window[1] = abs(int(self.entry2.get()))
            if window[0] == 0:
                window[0] = 5
            if window[1] == 0:
                window[1] = 5
        except:
            window[0] = 5
            window[1] = 5
        applyfilter = True
        self.alive = False
        tk.Tk.destroy(self)
        del self
        

def gaussian_window(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def low_pass_filter():
    global IM_process
    global applyfilter
    global window
    window = [5,5]
    try:
        if not root5.alive:
            root5 = lpf_image()
            root5.mainloop()
    except NameError:
        root5 = lpf_image()
        root5.mainloop()    
    if applyfilter:
        IM_process = convolve2d(IM_process, gaussian_window((window[0],window[1])), boundary='fill', mode='same')
        ax = plt.gca()
        ax.set_title('Low pass filter applied')

"""
7. Main processing loop
"""

IM_process = copy.deepcopy(IM_raw)
im_process = copy.deepcopy(im_raw)

fig2, ax2 = plt.subplots()
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='2%', pad=0.1)
im2 = ax2.imshow(IM_process,extent=[0,59.75,0,59.5],cmap=plt.get_cmap('Greys'),clim=[-10,10],interpolation='none',origin='lower')
ax2.set_title('Original Image')
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
cax.set_title('$\Delta$B [nT]')
cursor = Cursor(ax2, useblit=True, color='red', linewidth=2)
cursor.set_active(False)
fig2.canvas.mpl_connect('button_press_event', onclick)
fig2.canvas.mpl_connect('key_press_event', onkey)
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.draw()
curon = False
figbreak = False
grid = 1
shift = 3
fig.colorbar(im2, cax=cax,  orientation='vertical')
while plt.fignum_exists(1):
    if figbreak:
        figbreak = False
        break
    if curon:
        cursor.set_active(True)
    else:
        cursor.set_active(False)
    im2 = ax2.imshow(IM_process,extent=[0,59.75,0,59.75],cmap=plt.get_cmap('Greys'),clim=[-10,10],interpolation='none',origin='lower')
    plt.sca(ax2)
    plt.waitforbuttonpress()
    
plt.close(1)

sys.exit()

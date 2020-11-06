import tkinter
import PIL.Image, PIL.ImageTk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import os
#from B_box2 import my_Buttonbox
import cv2
import pyhdust.images as phim
import glob
from skimage import filters as fl


def compute_resize_factor(h,w,win_h,win_w):
  print(h)
  print(win_h)
  rf=(0.85*win_h)/h
  while True:
    rf -= 0.01
    if rf*w<(0.95*win_w):
        print(rf)
        return rf
import matplotlib.pyplot as plt
class App1:
 def __init__(self, window, window_title):
     self.window = window
     self.window.title(window_title)
     self.widthpixels = self.window.winfo_screenwidth()
     self.heightpixels = self.window.winfo_screenheight()
     print(self.heightpixels,  self.widthpixels)
     self.resize_factor = None
     self.window.geometry('{}x{}'.format(self.widthpixels, self.heightpixels))
     self.button_frame = Frame(self.window)
     self.button_frame.pack(side=BOTTOM, fill=Y)
     self.button_frame2 = Frame(self.window)
     self.button_frame2.pack(side=LEFT, fill=Y)
     self.img_frame = Frame(self.window)
     self.img_frame.pack(anchor=tkinter.CENTER, expand=True)
     labelframe1 = LabelFrame(self.button_frame, text='')
     labelframe1.pack(fill="both", expand="yes")
     self.labelframe3 = LabelFrame(self.window, text='')
     self.labelframe3.pack(side=BOTTOM, fill=Y)
     labelframe2 = LabelFrame(self.button_frame2, text='Working panel')
     labelframe2.pack(fill="both", expand="yes")
     self.choose_button = Button(labelframe2, text='import folder', height=2, width=10, command=self.select)
     self.choose_button.grid(row=0, column=0)
     self.start_button = Button(labelframe2, text='start', height=2, width=10, command=self.start)
     self.start_button.grid(row=1, column=0)
     self.start_button.config(state="disabled")
     self.prev_button = Button(labelframe1, text='Prev', height=1, width=10,command=self.prev)
     self.prev_button.grid(row=0, column=0)
     self.prev_button.config(state="disabled")
     self.dec_thresh_button0 = Button(labelframe1, text='-20', height=1, width=10, command=self.dec_thresh0)
     self.dec_thresh_button0.grid(row=0, column=1)
     self.dec_thresh_button0.config(state="disabled")
     self.dec_thresh_button = Button(labelframe1, text='-10', height=1, width=10, command=self.dec_thresh)
     self.dec_thresh_button.grid(row=0, column=2)
     self.dec_thresh_button.config(state="disabled")
     self.dec_thresh_button1 = Button(labelframe1, text='-5', height=1, width=10, command=self.dec_thresh1)
     self.dec_thresh_button1.grid(row=0, column=3)
     self.dec_thresh_button1.config(state="disabled")
     self.dec_thresh_button2 = Button(labelframe1, text='-1', height=1, width=10, command=self.dec_thresh2)
     self.dec_thresh_button2.grid(row=0, column=4)
     self.dec_thresh_button2.config(state="disabled")
     self.add_thresh_button2 = Button(labelframe1, text='+1', height=1, width=10, command=self.inc_thresh2)
     self.add_thresh_button2.grid(row=0, column=5)
     self.add_thresh_button2.config(state="disabled")
     self.add_thresh_button1 = Button(labelframe1, text='+5', height=1, width=10, command=self.inc_thresh1)
     self.add_thresh_button1.grid(row=0, column=6)
     self.add_thresh_button1.config(state="disabled")
     self.add_thresh_button = Button(labelframe1, text='+10', height=1, width=10, command=self.inc_thresh)
     self.add_thresh_button.grid(row=0, column=7)
     self.add_thresh_button.config(state="disabled")
     self.add_thresh_button0 = Button(labelframe1, text='+20', height=1, width=10, command=self.inc_thresh0)
     self.add_thresh_button0.grid(row=0, column=8)
     self.add_thresh_button0.config(state="disabled")
     self.next_button = Button(labelframe1, text='Next', height=1, width=10,command=self.next)
     self.next_button.grid(row=0, column=9)
     self.next_button.config(state="disabled")
     self.Failed_button = Button(labelframe2, text='Failed', height=2, width=10,command=self.Failed_fun)
     self.Failed_button.grid(row=4, column=0)
     self.Failed_button.config(state="disabled")
     self.cancel_button = Button(labelframe2, text='cancel', height=2, width=10,command=self.cancel)
     self.cancel_button.grid(row=3, column=0)
     self.cancel_button.config(state="disabled")











 def close_help(self):
     self.helpmaster.destroy()



 def get_frame(self,frame_number):
     if frame_number>=0 and frame_number<self.frame_num:
         # img = PIL.Image.open(self.frames[frame_number])
         # print(self.frames[frame_number])
         # double_size = (int(img.size[0] * self.resize_factor), int(img.size[1] * self.resize_factor))
         # img1 = np.asanyarray(img.resize(double_size))
         img1 = cv2.imread(self.frames[frame_number])
         img1 = cv2.resize(img1, dsize=(575,575))
         return (img1,True)
     else:
         return (None,False)





 def load_all(self,flag=False):
     for child in self.canvas.winfo_children():
         child.destroy()
     for child in self.canvas_mask.winfo_children():
         child.destroy()
     img1, ret = self.get_frame(self.frame_counter)
     img=img1.copy()
     self.out_folder = self.video_folder + '/' + self.frame_names[self.frame_counter]
     if os.path.exists(self.out_folder+'/'+'thresh.npy') and flag:
         self.thresh=np.load(self.out_folder+'/'+'thresh.npy')
     mask=self.seg_nucleus(img,self.thresh)
     mask2_0=img[:,:,0]
     mask2_0[mask>0]=255
     mask2_1 = img[:, :, 1]
     mask2_1[mask > 0] = 255
     mask2_2 = img[:, :, 2]
     mask2_2[mask > 0] = 255
     mask2=img
     mask2[:,:,0]=mask2_0
     mask2[:, :, 1] = mask2_1
     mask2[:, :, 2] = mask2_2
     self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)))
     self.photo_mask = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(mask2,cv2.COLOR_BGR2RGB)))
     self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
     self.canvas_mask.create_image(0, 0, image=self.photo_mask, anchor=tkinter.NW)

     if os.path.exists(self.out_folder) == False:
         os.makedirs(self.out_folder)






 def Failed_fun(self):
     if os.path.exists(self.out_folder) == False:
         os.makedirs(self.out_folder)
     np.save(self.out_folder+'/'+'thresh.npy',-1)







 def seg_nucleus(self,temp_image,thresh):

     #org_img = temp_image.copy()
     #temp_img = temp_image[100:475 , 100:475 , :]
     temp_img = temp_image.copy()
     output_shape = (len(temp_image[:,0,0]) , len(temp_image[0,:,0]) )
     #print('output shape : ' , output_shape)
     output_img = np.zeros(output_shape)
     #print(temp_image.shape)
     #temp_img=temp_image
     Gray = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
     B = temp_img[:, :, 0]
     G = temp_img[:, :, 1]
     R = temp_img[:, :, 2]

     mean_gray = np.mean(Gray)
     mean_R = np.mean(R)
     mean_G = np.mean(G)
     mean_B = np.mean(B)
     R_ = R * (mean_gray / mean_R)
     G_ = G * (mean_gray / mean_G)
     B_ = B * (mean_gray / mean_B)

     B_[B_>255] = 255
     G_[G_>255] = 255
     R_[R_>255] = 255
     balance_img = temp_img.copy()
     balance_img[:, :, 0] = B_.copy()
     balance_img[:, :, 1] = G_.copy()
     balance_img[:, :, 2] = R_.copy()

     balance_img = cv2.cvtColor(balance_img, cv2.COLOR_BGR2RGB)

     CMYK = phim.rgb2cmyk(np.asanyarray(balance_img))
     _M = CMYK[:, :, 1]

     ret_b, balance_bin = cv2.threshold(_M, thresh, 255, cv2.THRESH_BINARY)
     _, contours, _ = cv2.findContours(balance_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     # 9999999999999999999999999
     # Area = []
     # idx_Zone = []
     # tmp_Zone = []
     # pad_del = np.zeros_like(balance_bin)
     # print('*******************')
     # for j in range(len(contours)):
     #
     #     M = cv2.moments(contours[j])
     #     if M['m00'] != 0:
     #         Cx = int(M['m10'] / M['m00'])
     #         Cy = int(M['m01'] / M['m00'])
     #         value = balance_bin[Cy, Cx]
     #         if Cx > 0 and Cx < 575 and Cy > 0 and Cy <575:
     #             tmp_Zone.append(j)
     #             idx_Zone.append(j)
     #             Area.append(cv2.contourArea(contours[j]))
     #
     # print('area is : ' , Area)
     # print('shape of area is : ' , np.shape(Area))
     # max_area = np.max(Area)
     # for idx in tmp_Zone:
     #     if cv2.contourArea(contours[idx]) < (max_area / 10):
     #         idx_Zone.remove(idx)
     # print('idx zone is : ' , idx_Zone)
     # for idx_cnt in range(len(contours)):
     #     flag = True
     #     for elm in idx_Zone:
     #         if idx_cnt == elm:
     #             flag = False
     #     if flag:
     #         cv2.drawContours(pad_del, contours, idx_cnt, color=255, thickness=-1)
     #
     # balance_bin = balance_bin - pad_del
     # 999999999999999999999999999999
     # cv2.imshow('balance bin ' , balance_bin)
     # cv2.imshow('output img befor' , output_img)
     #output_img[100:475 , 100:475] = balance_bin.copy()
     #output_img[100:475 , 100:475] = balance_bin
     # cv2.imshow('output img after' , output_img)

     # cv2.waitKey(500)
     #output_img = cv2.resize(output_img,(500,500))
     return balance_bin


 def start(self):

     self.start_button.config(state="disabled")
     self.choose_button.config(state="disabled")
     for child in self.img_frame.winfo_children():
         child.destroy()
     self.tracker_num=0
     self.Icons=[]
     self.rect_xy=[]
     self.cell_mid=[]
     self.add_selected = False
     self.is_right=False
     #self.window_size=100
     self.next_button.config(state="normal")
     self.Failed_button.config(state="normal")
     self.prev_button.config(state="normal")
     self.add_thresh_button.config(state="normal")
     self.dec_thresh_button.config(state="normal")
     self.add_thresh_button1.config(state="normal")
     self.dec_thresh_button1.config(state="normal")
     self.add_thresh_button2.config(state="normal")
     self.dec_thresh_button2.config(state="normal")
     self.add_thresh_button0.config(state="normal")
     self.dec_thresh_button0.config(state="normal")
     self.cancel_button.config(state="normal")
     self.prev_label=0
     if os.path.exists('output') == False:
         os.makedirs('output')
     out_name = self.video_source.split('/')[-1]
     print('out_name : ' , out_name)

     self.frame_counter = -1
     if os.path.exists('output/' + out_name) == True:
         segmented_images = glob.glob('output/' + out_name +'/' + '*.jpg')
         print('segmented images : ' , segmented_images[0])
         self.frame_counter = len(segmented_images) - 2

     if os.path.exists('output/' + out_name) == False:
         os.makedirs('output/' + out_name)


     matches = []
     names=[]

     print('video source : ' , self.video_source)
     for root, dirnames, filenames in os.walk(self.video_source):
         #print('root' , root)
         #print('dirnames : ' , dirnames)
         #print('filenames : ' , filenames)
         for filename in filenames:
             if filename.endswith(".jpg"):
                 matches.append(os.path.join(root, filename))
                 names.append(filename)

     #print('names' , names)
     #print('matches : ' , matches)
     self.frames=matches
     self.frame_names=names
     self.frame_num=len(matches)
     self.video_folder='output/' + out_name
     img =np.asanyarray( PIL.Image.open(matches[0]))
     self.width=np.size(img,1)
     self.height=np.size(img,0)
     #self.resize_factor = compute_resize_factor(self.height, self.width, self.heightpixels, self.widthpixels)
     self.resize_factor=1
     self.canvas = tkinter.Canvas(self.img_frame, width=int(self.width*self.resize_factor), height=int(self.height*self.resize_factor))
     self.canvas_mask = tkinter.Canvas(self.img_frame, width=int(self.width * self.resize_factor),
                                  height=int(self.height * self.resize_factor))
     self.canvas.grid(row=0, column=0)
     self.canvas_mask.grid(row=0, column=1)
     self.thresh=120
     self.next()



 def click_bar(self,event,k):
     print('click:'+str(k))


 def select(self):
     for child in self.img_frame.winfo_children():
         child.destroy()
     #self.filename = filedialog.askopenfilename(initialdir="/home", title="Select a file",
     #                                           filetypes=(("Video files", "*.mp4"), ("all files", "*.*")))
     dirr = os.getcwd()
     self.filename = filedialog.askdirectory(initialdir=dirr,
                                             title="Select a folder",
                                             )

     print('selft.filename' , self.filename)
     if self.filename!=None:

         self.start_button.config(state="normal")
         self.video_source = self.filename



 def next(self):

     self.frame_counter+=1
     if self.frame_counter==self.frame_num:
         messagebox.showinfo("End of Images :", "the next images will be from first ")
         self.frame_counter=0

     print('image' , self.frame_counter , 'from ', len(self.frames) , ' images')

     img,ret=self.get_frame(self.frame_counter)

     Gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     B = img[:, :, 0]
     G = img[:, :, 1]
     R = img[:, :, 2]

     mean_gray = np.mean(Gray)
     mean_R = np.mean(R)
     mean_G = np.mean(G)
     mean_B = np.mean(B)
     R_ = R * (mean_gray / mean_R)
     G_ = G * (mean_gray / mean_G)
     B_ = B * (mean_gray / mean_B)

     B_[B_ > 255] = 255
     G_[G_ > 255] = 255
     R_[R_ > 255] = 255
     balance_img = img.copy()
     balance_img[:, :, 0] = R_.copy()
     balance_img[:, :, 1] = G_.copy()
     balance_img[:, :, 2] = B_.copy()

     CMYK = phim.rgb2cmyk(np.asanyarray(balance_img))
     _M = CMYK[:, :, 1]
     thresh = fl.threshold_multiotsu(_M, 2)
     self.thresh = thresh
     print('thresh is : ', self.thresh)

     if ret:
         self.load_all(True)
     else:
         self.frame_counter -= 1

 def prev(self):
     self.frame_counter-=1
     print('image', self.frame_counter, 'from ', len(self.frames), ' images')
     img,ret=self.get_frame(self.frame_counter)

     Gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     B = img[:, :, 0]
     G = img[:, :, 1]
     R = img[:, :, 2]

     mean_gray = np.mean(Gray)
     mean_R = np.mean(R)
     mean_G = np.mean(G)
     mean_B = np.mean(B)
     R_ = R * (mean_gray / mean_R)
     G_ = G * (mean_gray / mean_G)
     B_ = B * (mean_gray / mean_B)

     B_[B_>255] = 255
     G_[G_>255] = 255
     R_[R_>255] = 255
     balance_img = img.copy()
     balance_img[:, :, 0] = R_.copy()
     balance_img[:, :, 1] = G_.copy()
     balance_img[:, :, 2] = B_.copy()

     CMYK = phim.rgb2cmyk(np.asanyarray(balance_img))
     _M = CMYK[:, :, 1]
     thresh = fl.threshold_multiotsu(_M, 2)
     self.thresh = thresh
     print('thresh is : ', self.thresh)

     if ret:
         self.load_all(True)
     else:
         self.frame_counter += 1
 def inc_thresh(self):
     self.thresh=self.thresh+10
     if self.thresh>255:
         self.thresh=255
     self.load_all()
 def dec_thresh(self):
     self.thresh=self.thresh-10
     if self.thresh<0:
         self.thresh=0
     self.load_all()
 def inc_thresh1(self):
     self.thresh=self.thresh+5
     if self.thresh>255:
         self.thresh=255
     self.load_all()
 def dec_thresh1(self):
     self.thresh=self.thresh-5
     if self.thresh<0:
         self.thresh=0
     self.load_all()
 def inc_thresh2(self):
     self.thresh=self.thresh+1
     if self.thresh>255:
         self.thresh=255
     self.load_all()
 def dec_thresh2(self):
     self.thresh=self.thresh-1
     if self.thresh<0:
         self.thresh=0
     self.load_all()
 def inc_thresh0(self):
     self.thresh=self.thresh+20
     if self.thresh>255:
         self.thresh=255
     self.load_all()
 def dec_thresh0(self):
     self.thresh=self.thresh-20
     if self.thresh<0:
         self.thresh=0
     self.load_all()


 def cancel(self):
     for child in self.img_frame.winfo_children():
         child.destroy()
     for child in self.labelframe3.winfo_children():
         child.destroy()
     self.prev_button.config(state="disabled")
     self.next_button.config(state="disabled")
     self.Failed_button.config(state="disabled")
     self.add_thresh_button.config(state="disabled")
     self.dec_thresh_button.config(state="disabled")
     self.add_thresh_button1.config(state="disabled")
     self.dec_thresh_button1.config(state="disabled")
     self.add_thresh_button2.config(state="disabled")
     self.dec_thresh_button2.config(state="disabled")
     self.add_thresh_button0.config(state="disabled")
     self.dec_thresh_button0.config(state="disabled")
     self.cancel_button.config(state="disabled")
     self.start_button.config(state="normal")
     self.choose_button.config(state="normal")





 def on_closing(self):

     self.window.destroy()



a=App1(tkinter.Tk(),'Control Asabi-detecting Grond Truth App')
a.window.mainloop()

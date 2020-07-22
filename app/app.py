'''
Simple GUI to facilitate interaction with Fawkes.
'''

import threading
from tkinter import Tk, BOTH, StringVar, Canvas, PhotoImage, CENTER, NW
from tkinter.filedialog import askopenfilenames
from tkinter.ttk import Frame, Label, Button
from PIL import ImageTk, Image

import fawkes.protection


class UI(Frame):
    def __init__(self):
        super().__init__()
        self.my_fawkes = fawkes.protection.Fawkes("high_extract", '0', 1)
        self.var = StringVar()
        self.var.set('Select images to cloak!')
        self.img_paths = './imgs'
        self.initUI()

    def initUI(self):
        self.master.title("Fawkes")
        self.master.configure(bg='white')
        self.pack(fill=BOTH, expand=1)

        # fawkes image
        canvas = Canvas(self, width=110, height=150)
        orig = Image.open("fawkes_mask.jpg")
        resized = orig.resize((110,150), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(resized)
        canvas.create_image(0,0, image=img, anchor=NW)
        canvas.image = img
        canvas.pack()
        

        # open button
        btn_Open = Button(self,
                          text='Choose image(s) to cloak',
                          width=25,
                          command=self.select_path)
        btn_Open.pack()

        # run button
        btn_Run = Button(self,
                         text='Cloak images',
                         width=25,
                         command=lambda: thread_it(self.my_fawkes.run_protection, self.img_paths))
        btn_Run.pack()

        # # save button
        # btn_Save = Button(self,
        #                   text='Save cloaked image(s)',
        #                   width=25,
        #                   command=self.save_images)
        # btn_Save.pack()
        
        # Progress info
        Label_Show = Label(self,
                           textvariable=self.var,
                           font=('Arial', 13), width=50)
        Label_Show.configure(anchor="center")
        Label_Show.pack()
        

    def select_path(self):
        self.img_paths = askopenfilenames(filetypes=[('image', "*.gif *.jpg *.png")])
        self.var.set('Images chosen.')

    def save_images(self):
        print(self.img_paths)




root = Tk()
root.title('window')
root.geometry('200x230')
app = UI()


def main():
    root.configure(bg='white')
    root.mainloop()

def thread_it(func, *args):
    app.var.set('Cloaking in progress.')
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()
    while t.is_alive():
        pass
    app.var.set('Cloaking finished.')

def thread_test():
    app.var.set('Cloaking in progress.')

    def func(test):
        print(test)
    args = "testing"
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()
    while t.is_alive():
        pass
    t.sleep(1)
    app.var.set('Cloaking finished.')


if __name__ == '__main__':
    main()

import threading
from tkinter import Tk, BOTH, StringVar
from tkinter.filedialog import askopenfilenames
from tkinter.ttk import Frame, Label, Button

import fawkes.protection


class UI(Frame):
    def __init__(self):
        super().__init__()
        self.my_fawkes = fawkes.protection.Fawkes("high_extract", '0', 1)
        self.var = StringVar()
        self.var.set('Initial')
        self.img_paths = './imgs'
        self.initUI()

    def initUI(self):
        self.master.title("This is a Window")
        self.pack(fill=BOTH, expand=1)

        btn_Open = Button(self,
                          text='open img directory',
                          width=30,
                          command=self.select_path)
        btn_Open.pack()

        btn_Run = Button(self,
                         text='run the code',
                         width=3,
                         command=lambda: thread_it(self.my_fawkes.run_protection, self.img_paths))
        btn_Run.pack()

        Label_Show = Label(self,
                           textvariable=self.var,
                           font=('Arial', 13), width=50)
        Label_Show.pack()

    def select_path(self):
        self.img_paths = askopenfilenames(filetypes=[('image', "*.gif *.jpg *.png")])
        self.var.set('the paths have been set')


root = Tk()
root.title('window')
root.geometry('600x500')
app = UI()


def main():
    root.mainloop()


def thread_it(func, *args):
    app.var.set('cloak in process')
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()


if __name__ == '__main__':
    main()

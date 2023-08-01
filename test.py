from fawkes.protection import Fawkes
import os

fwks = Fawkes("extractor_2", '0', 1, mode="low")
fwks.run_protection(['./photos/P2.jpg'], format='jpeg')
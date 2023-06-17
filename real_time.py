import tkinter as tk
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from time import time
import numpy as np
import pandas as pd
from tkinter import filedialog
from tkinter import messagebox
#from tkcalendar import DateEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
def show_help():
     # Create a new Toplevel window to display the help message
    help_window = tk.Toplevel(root)

    # Add a label with the help message to the help window
    help_label = ttk.Label(help_window, text="""
Instructions on how to use the application:

1. Upload Data: You can upload EMG signals from your computer by clicking on the 'Upload File' button.

2. Configure Model Settings: The interface has an entry for name, user name, and date of birth to have a subject-specific analysis and prediction of your EMG signal. You can choose the number of movements and the number of repetitions you want to feed into your database. Note that if the number is low, you will get lower accuracy. You can select the movements from a dropdown menu in the parameter frame.

3. Interpret Results: You can visualize the EMG signal and filtered signal in real-time, start and stop the acquisition, and get the output of the prediction shown with an image. 

4. Modes: You can choose mode 1 to get the database, and mode 2 to predict. Note that mode 2 needs to have mode 1 model ready and uploaded to the interface.

""")
    help_label.pack(padx=10, pady=10)

# Define a function to handle the About button click event
def show_about():
     # Create a new Toplevel window to display the about message
    about_window = tk.Toplevel(root)

    # Add a label with the about message to the about window
    about_label = ttk.Label(about_window, text="""Welcome to the machine learning application! This interface is designed to serve two purposes: data collection and real-time movement prediction.

The first purpose of the application is to collect data and train a machine learning model using that data. You can use the application to upload and process EMG signals, configure the model settings, and train the model on the data you have collected.

Once the model is trained, the second purpose of the application is to use the trained model to predict movement in real time. You can upload EMG signals from a connected device, filter the signals, and visualize the output in real time. The predicted movement can be displayed as an image or other output, depending on the specific model you have trained.

We hope that this application will provide a powerful tool for understanding and analyzing EMG signals, and we are committed to providing comprehensive support and resources to help you make the most of its capabilities""")
    about_label.pack(padx=10, pady=10)

class state:
    def __init__(self, parent_frame):
         
        self.parent_frame = parent_frame
        # Create signal selection widgets
        frame1 = ttk.LabelFrame(self.parent_frame, text="Acquisition state", padding=(20, 10))
        frame1.grid(row=0, column=0, padx=10, pady=10)
        frame = ttk.LabelFrame(self.parent_frame, text="Acquisition state", padding=(20, 10))
        frame.grid(row=0, column=1, padx=10, pady=10)

        self.label = ttk.Label(frame1, text="Acquisition stopped", foreground="red")
        self.label.pack(padx=5, pady=5)

        start_button = ttk.Button(frame, text="Start", command=self.start)
        start_button.pack(padx=5, pady=5)

        stop_button = ttk.Button(frame, text="Stop", command=self.stop)
        stop_button.pack(padx=5, pady=5)
        
    def start(self):
        self.label.config(text="Acquisition on", foreground="green")
        
    def stop(self):
        self.label.config(text="Acquisition stopped", foreground="red")

root = tk.Tk()
root.title("Interface Classification Temps réel")
root.option_add("*tearOff", False) # This is always a good idea
#------------------------------------------------------------------------------------------------------------------------
# Make the app responsive
root.columnconfigure(index=0,weight=0)
root.columnconfigure(index=1, weight=2)
root.columnconfigure(index=2, weight=1)
root.rowconfigure(index=0, weight=1)
root.rowconfigure(index=1, weight=1)
root.rowconfigure(index=2, weight=1)

#------------------------------------------------------------------------------------------------------------------
# Here will set the theme color/change dark/light to get the dark and light mode (don't forget to change the graphs too)
style = ttk.Style(root)
root.tk.call("source", "C:/Users/massy/OneDrive/Bureau/zyada3alayha/Forest-ttk-theme-master/forest-dark.tcl")
style.theme_use("forest-dark")
#------------------------------------------------------------------------------------------------------------------

#---------------------------------------------------
# Create the menu bar
menu_bar = Menu(root)
root.config(menu=menu_bar)

file_menu = Menu(menu_bar)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open")

file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

settings_menu = Menu(menu_bar)
menu_bar.add_cascade(label="Settings", menu=settings_menu)
settings_menu.add_command(label="Change Theme")

help_menu = Menu(menu_bar)
menu_bar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About")

# ------------------------------------------------------------------------------------------------------------------
# Parameters frame
parameters_frame = ttk.LabelFrame(root, text="Parameters", padding=(10, 10))
parameters_frame.grid(column=0, padx=(10, 10), pady=10, sticky="nsew", rowspan=3)
#info frame
info_frame = ttk.LabelFrame(parameters_frame, text="Information", padding=(10, 10))
info_frame.grid(row=1, column=0, padx=5, pady=5, sticky="w")
#Add the name and surname entry fields
name_label = ttk.Label(info_frame, text="Subject number", padding=(5, 5))
name_label.grid(row=1, column=0, sticky="w")
name_entry = ttk.Entry(info_frame)
name_entry.grid(row=1, column=1, sticky="we", padx=5, pady=5)

surname_label = ttk.Label(info_frame, text="Subject age", padding=(5, 5))
surname_label.grid(row=2, column=0, sticky="w")
surname_entry = ttk.Entry(info_frame)
surname_entry.grid(row=2, column=1, sticky="we", padx=5, pady=5)

#-----------------------------------------------------------------------------------------------------------------

  #Add the repetitions and movements options in a separate frame
repmov_frame = ttk.LabelFrame(parameters_frame, text="Repetitions and Movements", padding=(10, 10))
repmov_frame.grid(row=2, column=0, padx=5, pady=5, sticky="w")

repetitions_label = ttk.Label(repmov_frame, text="Number of Repetitions:", padding=(5, 5))
repetitions_label.grid(row=0, column=0, sticky="w")
repetitions_var = tk.StringVar()
repetitions_menu = ttk.OptionMenu(repmov_frame, repetitions_var,"1", "1", "2", "3", "4", "5", "6", "7")
repetitions_menu.grid(row=0, column=1, sticky="we", padx=5, pady=5)

movements_label = ttk.Label(repmov_frame, text="Number of Movements:", padding=(5, 5))
movements_label.grid(row=1, column=0, sticky="w")
movements_var = tk.StringVar()
movements_menu = ttk.OptionMenu(repmov_frame, movements_var,"1", "1", "2", "3", "4", "5", "6","7")
movements_menu.grid(row=1, column=1, sticky="we", padx=5, pady=5)
#------------------------------------------------------------------------------------------------------------------
# raw signal parameters frame
raw_signal_frame = ttk.LabelFrame(parameters_frame, text="Raw Signal Parameters", padding=(10, 10))
raw_signal_frame.grid(row=3, column=0, padx=5, pady=5, sticky="w")
#  electrode selection dropdown
electrode_label = ttk.Label(raw_signal_frame, text="Number of Electrodes:", padding=(5, 5))
electrode_label.grid(row=0, column=0, sticky="w")
electrode_variable = tk.StringVar()
electrode_dropdown = ttk.OptionMenu(raw_signal_frame, electrode_variable,"One Electrode", "One Electrode", "Two Electrodes")
electrode_dropdown.grid(row=0, column=1, sticky="we", padx=5, pady=5)
#signal color selection dropdown
color_label = ttk.Label(raw_signal_frame, text="Signal Color Display:")
color_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
color_variable = tk.StringVar(value="SELECT COLOR")
color_dropdown = ttk.OptionMenu(raw_signal_frame, color_variable, "Red","Red", "Green", "Blue")
color_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="we")


#------------------------------------------------------------------------------------------------------------------
# # Filtred signal parameters frame
filtred_signal_frame = ttk.LabelFrame(parameters_frame, text="filtred Signal Parameters", padding=(20, 20))
filtred_signal_frame.grid(row=4, column=0, padx=5, pady=5, sticky="w")
 #filter selection dropdown
filter_label = ttk.Label(filtred_signal_frame, text="Filter Used:")
filter_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
filter_variable = tk.StringVar(value="SELECT FILTER")
filter_dropdown = ttk.OptionMenu(filtred_signal_frame, filter_variable, "Filter A","Filter A", "Filter B", "Filter C")
filter_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="we")

 #signal color selection dropdown
color_label = ttk.Label(filtred_signal_frame, text="Signal Color Display:")
color_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
color_variable = tk.StringVar(value="SELECT COLOR")
color_dropdown = ttk.OptionMenu(filtred_signal_frame, color_variable, "Red","Red", "Green", "Blue")
color_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="we")



#------------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------
# Create the Help button and add it to the frame
help_button = ttk.Button(parameters_frame, text="Help", command=show_help)
help_button.grid(row=7, column=0, padx=5, pady=5)

# Create the About button and add it to the frame
about_button = ttk.Button(parameters_frame, text="About", command=show_about)
about_button.grid(row=6, column=0, padx=5, pady=5)
#------------------------------------------------------------------------------------------------------------------
# frame for the signal (raw signal)
signalRaw_frame = ttk.LabelFrame(root, text="   Raw Signal", padding=(10, 10))
signalRaw_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
fig, ax = plt.subplots(figsize=(5, 4), dpi=100)  # Set the facecolor to black
canvas = FigureCanvasTkAgg(fig, master=signalRaw_frame)
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


#------------------------------------------------------------------------------------------------------------------
# frame for filtred signal

filteredsignal_frame = ttk.LabelFrame(root, text="Filtered Signal", padding=(10, 10))
filteredsignal_frame.grid(row=1, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=100)  # Set the facecolor to black
canvas1 = FigureCanvasTkAgg(fig1, master=filteredsignal_frame)
canvas1.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


 #--------------------------------------
# frame for tome start/stop and state
state_frame = ttk.LabelFrame(root, text="Hand tracking", padding=(20, 10))
state_frame.grid(row=0, column=2, padx=(20, 10), pady=10, sticky="nsew")

#------------------------------------------------------------------------------------------------------
# frame for mouvement
mouvement_frame = ttk.LabelFrame(root, text="Mouvement", padding=(20, 10))
mouvement_frame.grid(row=1, column=2, padx=(20, 10), pady=10, sticky="nsew",rowspan=3)
image = Image.open("IMG_0078.JPG")
photo = ImageTk.PhotoImage(image.resize((400, 400))) 
label = ttk.Label(mouvement_frame, image=photo)
label.pack()

#------------------------------------------------------------------------------------------------------------------
# Center the window, and set minsize
root.update()
root.minsize(root.winfo_width(), root.winfo_height())
x_cordinate = int((root.winfo_screenwidth()/2) - (root.winfo_width()/2))
y_cordinate = int((root.winfo_screenheight()/2) - (root.winfo_height()/2))
root.geometry("+{}+{}".format(x_cordinate, y_cordinate))

# Start the main loop
root.mainloop()

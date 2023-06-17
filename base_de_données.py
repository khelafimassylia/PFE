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
import glob
import csv
#from tkcalendar import DateEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
global_variables = {}
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
    help_label.pack(padx=5, pady=5)
# Define a function to handle the About button click event
def show_about():
     # Create a new Toplevel window to display the about message
    about_window = tk.Toplevel(root)

    # Add a label with the about message to the about window
    about_label = ttk.Label(about_window, text="""Welcome to the machine learning application! This interface is designed to serve two purposes: data collection and real-time movement prediction.

The first purpose of the application is to collect data and train a machine learning model using that data. You can use the application to upload and process EMG signals, configure the model settings, and train the model on the data you have collected.

Once the model is trained, the second purpose of the application is to use the trained model to predict movement in real time. You can upload EMG signals from a connected device, filter the signals, and visualize the output in real time. The predicted movement can be displayed as an image or other output, depending on the specific model you have trained.

We hope that this application will provide a powerful tool for understanding and analyzing EMG signals, and we are committed to providing comprehensive support and resources to help you make the most of its capabilities""")
    about_label.pack(padx=5, pady=5)
class ImageDisplay:
    def __init__(self, parent_frame, repetitions, time_pause, duration):
        self.parent_frame = parent_frame
        self.image_paths = glob.glob('C:/Users/massy/OneDrive/Bureau/zyada3alayha/mouvement/*.jpg')
        self.image_paths = [path for path in self.image_paths if 'rest.jpg' not in path]
        self.pause_image = "C:/Users/massy/OneDrive/Bureau/zyada3alayha/mouvement/rest.jpg"
        self.repetitions = int(repetitions)
        self.time_pause = int(time_pause) *1000
        self.duration = int(duration) *1000
        self.current_image_index = 0
        self.paused = False
        self.photo = None
        self.image_label = tk.Label(self.parent_frame)
        self.image_label.pack()
        self.movement_label = tk.Label(self.parent_frame, text="", font=("Arial", 14))
        self.movement_label.pack(pady=10)
        self.translation_dict = {
            "1HC.jpg": "Hand close",
            "2HO.jpg": "Hand open",
            "3T.jpg": "Thumb down",
            "4I.jpg": "Index down",
            "5M.jpg": "Middle down",
            "6R.jpg": "Ring down",
            "7P.jpg": "Pinky down",
        }
        self.acquisition_on = False
    def display_image(self):
        if self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            image = Image.open(image_path)
            image = image.resize((300, 300))
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo)

            movement_name = self.get_movement_name(image_path)
            repetition_number = int(self.repetitions)
            self.movement_label.config(text=f"{movement_name} - Repetition {repetition_number}")

            if int(self.repetitions) > 1:
                self.parent_frame.after(5000, self.display_rest)
                self.repetitions = int(self.repetitions) - 1
            else:
                self.current_image_index += 1  # Move to the next image
                self.parent_frame.after(5000, self.display_image)
        else:
            # Display the last image or perform any required action after all images have been displayed
            self.current_image_index = 0

    def display_rest(self):
        image_pause = Image.open(self.pause_image)
        image_pause = image_pause.resize((300, 300))
        # self.photo = ImageTk.PhotoImage(image_pause)
        # self.image_label.config(image=self.photo)
        self.parent_frame.after(3000, self.display_image)

    def next_image(self):
        if self.acquisition_on:
            self.current_image_index += 1
            image_pause = Image.open(self.pause_image)
            image_pause = image_pause.resize((300, 300))
            self.photo = ImageTk.PhotoImage(image_pause)
            self.image_label.config(image=self.photo)
            self.parent_frame.after(3000, self.display_image)

    # def display_image(self):
    #     if self.current_image_index < len(self.image_paths):
    #         image_path = self.image_paths[self.current_image_index]
    #         image = Image.open(image_path)
    #         image = image.resize((300, 300))
    #         self.photo = ImageTk.PhotoImage(image)
    #         self.image_label.config(image=self.photo)

    #         movement_name = self.get_movement_name(image_path)
    #         repetition_number = int(self.repetitions)
    #         self.movement_label.config(text=f"{movement_name} - Repetition {repetition_number}")

    #         if int(self.repetitions) > 1:
    #             self.parent_frame.after(self.duration, self.display_rest)
    #             self.repetitions = int(self.repetitions) - 1
    #         else:
    #             self.parent_frame.after(self.duration, self.next_image)

    def display_rest(self):
        image_pause = Image.open(self.pause_image)
        image_pause = image_pause.resize((300, 300))
        self.photo = ImageTk.PhotoImage(image_pause)
        self.image_label.config(image=self.photo)
        self.parent_frame.after(self.time_pause, self.display_image)

    def next_image(self):
        if self.acquisition_on:
            self.current_image_index += 1
            image_pause = Image.open(self.pause_image)
            image_pause = image_pause.resize((300, 300))
            self.photo = ImageTk.PhotoImage(image_pause)
            self.image_label.config(image=self.photo)
            self.parent_frame.after(self.time_pause, self.display_image)

    def get_movement_name(self, image_path):
        # Extract movement name from the image file name
        image_file_name = image_path.split("\\")[-1]  # Get the file name from the image path
        movement_name = self.translation_dict.get(image_file_name, image_file_name)
        return movement_name

class State:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.image_display = None
        self.acquisition_on = False
        self.paused = False
        frame1 = ttk.LabelFrame(self.parent_frame, padding=(20, 10))
        frame1.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        frame = ttk.LabelFrame(self.parent_frame,text="Aquisition state" ,padding=(20, 10))
        frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
        self.label = ttk.Label(frame1, text="Acquisition stopped", foreground="red", font=("Arial", 32))
        self.label.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        self.start_button = ttk.Button(frame, text="Start", command=self.start)
        self.start_button.grid(row=1, column=0, padx=20, pady=20)
        self.pause_button = ttk.Button(frame, text="pause", command=self.toggle_pause)
        self.pause_button.grid(row=1, column=1, padx=20, pady=20)
        self.stop_button = ttk.Button(frame, text="Stop", command=self.stop)
        self.stop_button.grid(row=1, column=2, padx=20, pady=20)
         # Create the image frame
        image_test_frame = ttk.LabelFrame(root, text="Image Display", padding=(10, 10))
        image_test_frame.grid(row=1, column=2,padx=10, pady=10)
        image_frame = ttk.LabelFrame(image_test_frame, text="Mouvement image ", padding=(10, 10))
        image_frame.grid(row=1, column=2, padx=10, pady=10)


        # Create an instance of the ImageDisplay class
        self.image_display = ImageDisplay(image_frame,repetitions=global_variables.get('num_repetitions'),time_pause=global_variables.get('time_pause'),duration=global_variables.get('duration'))
    def start(self):
        if not self.acquisition_on:
            # Check if the required fields are filled
            if not global_variables.get('subject_number') or not global_variables.get('subject_age') or \
                    not global_variables.get('num_movements') or not global_variables.get('num_repetitions') or \
                    not global_variables.get('duration') or not global_variables.get('time_pause'):
                messagebox.showerror("Error", "Please fill in all the required fields and hit save variables to start the recording")
                return

            self.acquisition_on = True
            self.label.config(text="Acquisition on", foreground="green")
            if not self.image_display:
                num_repetitions = global_variables['num_repetitions']
                time_pause = global_variables['time_pause']*1000
                duration = global_variables['duration']*1000
                self.image_display = ImageDisplay(self.parent_frame, num_repetitions, time_pause, duration)
            # else:
            #     self.image_display.time_pause = global_variables['time_pause'] * 1000
            #     self.image_display.duration = global_variables['duration'] * 1000

            num_repetitions = global_variables['num_repetitions']
            self.image_display.repetitions = num_repetitions
            self.image_display.display_image()
                

    def stop(self):
            if not self.acquisition_on:
                messagebox.showerror("Error", "You can't stop the acquisition before starting it")
                return
            if self.acquisition_on:
                self.acquisition_on = False
                self.label.config(text="Acquisition stopped", foreground="red")
                
    def toggle_pause(self):
            if not self.acquisition_on:
                messagebox.showerror("Error", "You can't pause the acquisition without starting it")
                return      
            if self.paused:
                self.paused = False
                self.label.config(text="Acquisition on", foreground="green")
            else:
                self.paused = True
                self.label.config(text="Acquisition paused", foreground="orange")

def save_variables():
    global global_variables
    global_variables['subject_number'] = name_entry.get()
    global_variables['subject_age'] = surname_entry.get()
    global_variables['num_movements'] = movements_var.get()
    global_variables['num_repetitions'] = repetitions_var.get()
    global_variables['time_pause'] = repetitions_var.get()
    global_variables['duration'] = repetitions_var.get()

def browse_directory():
    directory = filedialog.askdirectory()
    directory_entry.delete(0, tk.END)
    directory_entry.insert(0, directory)

# def create_folders_and_files():
   
#     subject_number = global_variables.get('subject_number')
#     subject_age = global_variables.get('subject_age')
#     num_movements = int(global_variables.get('num_movements'))
#     num_repetitions = int(global_variables.get('num_repetitions'))


#     directory = directory_entry.get()

#     if directory:
   
#         folder_name = f"subject{subject_number}-age{subject_age}"
#         folder_path = os.path.join(directory, folder_name)
#         os.makedirs(folder_path, exist_ok=True)
#         for movement in range(1, num_movements + 1):
#             movement_folder = os.path.join(folder_path, f"Movement{movement}")
#             os.makedirs(movement_folder, exist_ok=True)
#             for repetition in range(1, num_repetitions + 1):
#                 csv_file = os.path.join(movement_folder, f"Repetition{repetition}.csv")
               
#                 with open(csv_file, 'w', newline='') as file:
#                     writer = csv.writer(file)
# #                     writer.writerow(["Column1", "Column2", "Column3"])
import os

def create_folders_and_files():
    subject_number = global_variables.get('subject_number')
    subject_age = global_variables.get('subject_age')
    num_movements = int(global_variables.get('num_movements'))
    num_repetitions = int(global_variables.get('num_repetitions'))

    directory = directory_entry.get()

    if directory:
        folder_name = f"subject{subject_number}-age{subject_age}"
        folder_path = os.path.join(directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        info_file = os.path.join(folder_path, "Information.txt")
        
        with open(info_file, 'w') as file:
            file.write(f"Subject Number: {subject_number}\n")
            file.write(f"Subject Age: {subject_age}\n")
            file.write(f"Number of Movements: {num_movements}\n")
            file.write(f"Number of Repetitions: {num_repetitions}\n")
        
        for movement in range(1, num_movements + 1):
            movement_folder = os.path.join(folder_path, f"Movement{movement}")
            os.makedirs(movement_folder, exist_ok=True)


root = tk.Tk()
root.title("Interface Classification")
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

def plot_signal_from_csv(directory, filename, ax):
    # Read the CSV file into a pandas DataFrame
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)

    # Get the signal data from the DataFrame
    signal = df.values

    # Plot the signal on the specified axes
    ax.plot(signal)

    # Get the selected filter and color
    selected_filter = filter_variable.get()
    #selected_color = color_variable.get()

    # Add labels and title to the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal Plot')

    # Customize the plot based on the selected options
    ax.grid(True)
    
    ax.legend([selected_filter], loc='upper right')
    #ax.set_facecolor(selected_color)

def plot_filtred_signal_from_csv(directory, filename, ax):
    
    # Read the CSV file into a pandas DataFrame
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)

    # Get the signal data from the DataFrame
    signal = df.values

    # Plot the signal on the specified axes
    ax.plot(signal)

    # Get the selected filter and color
    selected_filter = filter_variable.get()
    #selected_color = color_variable.get()

    # Add labels and title to the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal Plot')

    # Customize the plot based on the selected options
    ax.grid(True)
    ax.legend([selected_filter], loc='upper right')
    #ax.set_facecolor(selected_color)
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
global_variables['num_repetitions'] = repetitions_var.get()
global_variables['num_movements'] = movements_var.get()
#--------------------------------------------------------------------------------------------
#hna nregliw timing
timing_frame = ttk.LabelFrame(parameters_frame, text="Timing", padding=(10, 10))
timing_frame.grid(row=3, column=0, padx=5, pady=5, sticky="w")

duartion_label = ttk.Label(timing_frame, text="Duration of mouvement:", padding=(5, 5))
duartion_label.grid(row=0, column=0, sticky="w")
duration= tk.StringVar()
duartion_menu = ttk.OptionMenu(timing_frame, duration,"1", "1", "2", "3", "4", "5")
duartion_menu.grid(row=0, column=1, sticky="we", padx=5, pady=5)

movements_label = ttk.Label(timing_frame, text="Duration of pause", padding=(5, 5))
movements_label.grid(row=1, column=0, sticky="w")
time_pause= tk.StringVar()
movements_menu = ttk.OptionMenu(timing_frame, time_pause,"1", "1", "2", "3")
movements_menu.grid(row=1, column=1, sticky="we", padx=5, pady=5)
global_variables['duration'] = duration.get()
global_variables['time_pause'] = movements_var.get()
#-------------------------------------------------------------------------------------------
# Create a button to save the variables
savig_avr_frame = ttk.LabelFrame(parameters_frame, text="Saving", padding=(10, 10))
savig_avr_frame.grid(row=4, column=0, padx=5, pady=5, sticky="w")

saving_label = ttk.Label(savig_avr_frame, text="Saving the informations", padding=(5, 5))
saving_label.grid(row=0, column=0, sticky="w")

save_button = ttk.Button(savig_avr_frame, text="Save Variables", command=save_variables)
save_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")
#------------------------------------------------------------------------------------------------------------------
# raw signal parameters frame
raw_signal_frame = ttk.LabelFrame(parameters_frame, text="Raw Signal Parameters", padding=(10, 10))
raw_signal_frame.grid(row=5, column=0, padx=5, pady=5, sticky="w")
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
color_dropdown = ttk.OptionMenu(raw_signal_frame, color_variable, "Red","blue/orange", "Green", "Blue")
color_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="we")


#------------------------------------------------------------------------------------------------------------------
# # Filtred signal parameters frame
filtred_signal_frame = ttk.LabelFrame(parameters_frame, text="filtred Signal Parameters", padding=(20, 20))
filtred_signal_frame.grid(row=6, column=0, padx=5, pady=5, sticky="w")

 #filter selection dropdown
filter_label = ttk.Label(filtred_signal_frame, text="Filter Used:")
filter_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
filter_variable = tk.StringVar(value="SELECT FILTER")
filter_dropdown = ttk.OptionMenu(filtred_signal_frame, filter_variable, "Filter A","Notch", "Filter B", "Filter C")
filter_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="we")

 #signal color selection dropdown
color_label = ttk.Label(filtred_signal_frame, text="Signal Color Display:")
color_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
color_variable = tk.StringVar(value="SELECT COLOR")
color_dropdown = ttk.OptionMenu(filtred_signal_frame, color_variable, "Red","blue/orange", "Green", "Blue")
color_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="we")


#------------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------
# Create the Help button and add it to the frame
# help_button = ttk.Button(parameters_frame, text="Help", command=show_help)
# help_button.grid(row=7, column=0, padx=5, pady=5)

# # Create the About button and add it to the frame
# about_button = ttk.Button(parameters_frame, text="About", command=show_about)
# about_button.grid(row=7, column=1, padx=5, pady=5)
#------------------------------------------------------------------------------------------------------------------
# frame for the signal (raw signal)
signalRaw_frame = ttk.LabelFrame(root, text="   Raw Signal", padding=(10, 10))
signalRaw_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
fig, ax = plt.subplots(figsize=(5, 4), dpi=100)  # Set the facecolor to black
canvas = FigureCanvasTkAgg(fig, master=signalRaw_frame)
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

plot_button = ttk.Button(filtred_signal_frame, text="Plot Signal", command=lambda: plot_signal_from_csv("C:/Users/massy/OneDrive/Bureau/THE REAL PFE/said/HC", "HC_1.csv",ax))
plot_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

#------------------------------------------------------------------------------------------------------------------
# frame for filtred signal

filteredsignal_frame = ttk.LabelFrame(root, text="Filtered Signal", padding=(10, 10))
filteredsignal_frame.grid(row=1, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=100)  # Set the facecolor to black
canvas1 = FigureCanvasTkAgg(fig1, master=filteredsignal_frame)
canvas1.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
plot_button = ttk.Button(raw_signal_frame, text="Plot Signal", command=lambda: plot_filtred_signal_from_csv("C:/Users/massy/OneDrive/Bureau/THE REAL PFE/said/HC", "HC_1.csv",ax1))
plot_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

#------------------------------------------------------------------------------------------------------------------
# frame for tome start/stop and state

state_frame = ttk.LabelFrame(root, text="State", padding=(20, 10))
state_frame.grid(row=0, column=2, padx=(20, 10), pady=10, sticky="nsew")
 #adding the button that will activate start (timer /and aquisition state) yawedi nehi timer
state = State(state_frame)
button = ttk.Button(state_frame, text="Button", command=state.start)

#haya nzidou datasetframe
dataset_frame = ttk.LabelFrame(state_frame, text="Repetitions and Movements", padding=(10, 10))
dataset_frame.grid(row=2, column=0, padx=5, pady=5, sticky="w")


# Add the browse button and entry field
directory_label = ttk.Label(dataset_frame, text="Database Directory:", padding=(5, 5))
directory_label.grid(row=1, column=0, sticky="w")

directory_entry = ttk.Entry(dataset_frame)
directory_entry.grid(row=1, column=1, sticky="we", padx=5, pady=5)

browse_button = ttk.Button(dataset_frame, text="Browse", command=browse_directory)
browse_button.grid(row=1, column=2, padx=5, pady=5, sticky="e")

create_database_label = ttk.Label(dataset_frame, text="Create_database:", padding=(5, 5))
create_database_label.grid(row=3, column=0, sticky="w")

create_button = ttk.Button(dataset_frame, text="Create database", command=create_folders_and_files)
create_button.grid(row=3, column=1, padx=5, pady=5, sticky="e")

#---------------------------------------------------------------------------
# Create the image frame
# image_frame = ttk.LabelFrame(root, text="Image Display", padding=(10, 10))
# image_frame.grid(row=1, column=2,padx=10, pady=10)

# Create an instance of the ImageDisplay class
# image_display = ImageDisplay(image_frame)
# image_display.display_image()
# #------------------------------------------------------------------------------------------------------------------
# Center the window, and set minsize
root.update()
root.minsize(root.winfo_width(), root.winfo_height())
x_cordinate = int((root.winfo_screenwidth()/2) - (root.winfo_width()/2))
y_cordinate = int((root.winfo_screenheight()/2) - (root.winfo_height()/2))
root.geometry("+{}+{}".format(x_cordinate, y_cordinate))

# Start the main loop
root.mainloop()

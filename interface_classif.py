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
from backend_classif import classif
import threading
from alive_progress import alive_bar
from time import sleep
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
def browse_directory():
    folder_path = filedialog.askdirectory()
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)
    extract_information(folder_path)


def extract_information(folder_path):
    info_file = os.path.join(folder_path, "Information.txt")
    variables = []

    with open(info_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split the line by colon (:) and extract the variable name and value
            parts = line.split(':')
            if len(parts) == 2:
                variable_name = parts[0].strip()
                variable_value = parts[1].strip()
                variables.append((variable_name, variable_value))

    display_information(variables)

def display_information(variables):
    # Clear existing labels
    for child in info_frame.winfo_children():
        child.destroy()

    # Iterate over the variables and display them in the labels
    for i, variable in enumerate(variables):
        variable_name, variable_value = variable

        # Create a label for each variable
        label = ttk.Label(info_frame, text=f"{variable_name}: {variable_value}", padding=(5, 5),foreground="green")
        label.grid(row=i, column=0, sticky="w")

#----------------------------------------------------------------------------------------------------
def select_folder():
    folder_selected = filedialog.askdirectory()
    folder_path.set(folder_selected)
    get_csv_file_names()

selected_csv_file = ""
def get_csv_file_names():
    global selected_csv_file
    folder = folder_path.get()
    file_list = os.listdir(folder)
    csv_files = [file for file in file_list if file.endswith(".csv")]
    csv_names = list(set([file.split(".")[0][0:2] for file in csv_files]))
    csv_dropdown["menu"].delete(0, "end")
    for name in csv_names:
        csv_dropdown["menu"].add_command(label=name, command=lambda name=name: set_selected_csv_file(name))
    if csv_names:
        set_selected_csv_file(csv_names[0])

def set_selected_csv_file(name):
    global selected_csv_file
    selected_csv_file = name
    csv_variable.set(name)
#-----------------------------------------------------------------------------------------------------
def classify_folder():
    folder = folder_path.get()
    if not folder:
        messagebox.showwarning("Folder Not Selected", "Please select a folder.")
        return
    def classify_and_update_progress():
            progress['value'] = 0
            for _ in range(100):
                sleep(0.03)
                progress.step(1)
                root.update_idletasks()

            # Classification process
            classifiers, accuracy_train, accuracy_test, classifier_clf = classif(folder)
            confusion_matrices = []
            # for i in range(5):
            #     cm = confusion_matrix(y_true, y_pred)
            #     confusion_matrices.append(cm)

            # # Print confusion matrices
            # for i in range(5):
            #     matrix_label = tk.Label(metrics_frame, text=f"Confusion Matrix {i+1}:")
            #     matrix_label.grid(row=i+1, column=0, padx=5, pady=5, sticky="w")
            #     matrix_text = tk.Text(metrics_frame, width=50, height=5)
            #     matrix_text.insert(tk.END, str(confusion_matrices[i]))
            #     matrix_text.configure(state='disabled')
            #     matrix_text.grid(row=i+1, column=1, padx=5, pady=5, sticky="w")

            # Display the train and test accuracy results with star ratings and percentages
            train_labels = ["Train Accuracy:"]
            test_labels = ["Test Accuracy:"]
            for i, result in enumerate(accuracy_train):
                stars = int(result * 5)  # Scale accuracy to 5 stars
                rating = "★" * stars + "☆" * (5 - stars)  # Display stars and empty stars
                percentage = f"({result * 100:.2f}%)"
                train_labels.append(f"{classifiers[i]}: {rating} {percentage}")
            for i, result in enumerate(accuracy_test):
                stars = int(result * 5)  # Scale accuracy to 5 stars
                rating = "★" * stars + "☆" * (5 - stars)  # Display stars and empty stars
                percentage = f"({result * 100:.2f}%)"
                test_labels.append(f"{classifiers[i]}: {rating} {percentage}")

            # Update the labels in the main thread
            root.after(0, lambda: train_label.config(text='\n'.join(train_labels)))
            root.after(0, lambda: test_label.config(text='\n'.join(test_labels)))

        # Start the classification and update progress in the main thread
    
    root.after(0, classify_and_update_progress)
   

#-------------------------------------------------------
#----------------------------------------------------------------------------------------------------
root = tk.Tk()
root.title("Interface Classification ")
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

# ------------------------------------------------------------------------------------------------------------------
# Parameters frame
parameters_frame = ttk.LabelFrame(root, text="Parameters", padding=(20, 10))
parameters_frame.grid(column=0, padx=(20, 10), pady=10, sticky="nsew", rowspan=3)
folder_path = tk.StringVar()
csv_variable = tk.StringVar(value="Select a CSV file")
child_frame = ttk.Frame(parameters_frame)
child_frame.grid(column=0, row=0, padx=10, pady=10, sticky="nsew")

folder_label = ttk.Label(child_frame, text="Select a folder:")
folder_label.grid(row=0, column=0, pady=10)

folder_button = ttk.Button(child_frame, text="Browse", command=select_folder)
folder_button.grid(row=0, column=1, pady=10)

csv_label = ttk.Label(child_frame, text="Select a CSV file:")
csv_label.grid(row=1, column=0, pady=10)

csv_dropdown = ttk.OptionMenu(child_frame, csv_variable, "Select a CSV file")
csv_dropdown.grid(row=1, column=1, pady=10)


# # Add the browse button and entry field
# directory_label = ttk.Label(classif_folder_frame, text="Database Directory:", padding=(5, 5))
# directory_label.grid(row=1, column=0, sticky="w")

# directory_entry = ttk.Entry(classif_folder_frame)
# directory_entry.grid(row=1, column=0, sticky="we", padx=5, pady=5)

# browse_button = ttk.Button(classif_folder_frame, text="Browse", command=browse_directory)
# browse_button.grid(row=1, column=1, padx=5, pady=5, sticky="e")

#-----------------------------------------------------------------------------------------------------------------
#info frame

info_frame = ttk.LabelFrame(parameters_frame, text="Information", padding=(10, 10))
info_frame.grid(row=2, column=0, padx=5, pady=5, sticky="w")
name_label = ttk.Label(info_frame, text="Subject number:", padding=(5, 5))
name_label.grid(row=0, column=0, sticky="w")
surname_label = ttk.Label(info_frame, text="Subject age:", padding=(5, 5))
surname_label.grid(row=1, column=0, sticky="w")
num_mouv_label = ttk.Label(info_frame, text="Number of mouvement:", padding=(5, 5))
num_mouv_label.grid(row=2, column=0, sticky="w")
num_repetition_label = ttk.Label(info_frame, text="Number of repetition:", padding=(5, 5))
num_repetition_label.grid(row=3, column=0, sticky="w")
time_mouv_label = ttk.Label(info_frame, text="Time for each mouvement:", padding=(5, 5))
time_mouv_label.grid(row=4, column=0, sticky="w")
time_pause_label = ttk.Label(info_frame, text="Time pause :", padding=(5, 5))
time_pause_label.grid(row=5, column=0, sticky="w")



#-----------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
# raw signal parameters frame
raw_signal_frame = ttk.LabelFrame(parameters_frame, text="Raw Signal Parameters", padding=(10, 10))
raw_signal_frame.grid(row=4, column=0, padx=5, pady=5, sticky="w")
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
filtred_signal_frame.grid(row=5, column=0, padx=5, pady=5, sticky="w")
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
fig, ax = plt.subplots(figsize=(5, 4), dpi=100)  
canvas = FigureCanvasTkAgg(fig, master=signalRaw_frame)
canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


#------------------------------------------------------------------------------------------------------------------


 #--------------------------------------
# frame for tome start/stop and state
classif_frame = ttk.LabelFrame(root, text="Classification", padding=(20, 10))
classif_frame.grid(row=0, column=2, padx=(20, 10), pady=10, sticky="nsew")


train_frame = ttk.LabelFrame(classif_frame, text="training results", padding=(20, 10))
train_frame.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="nsew")
train_label = tk.Label(train_frame, text='', padx=10, pady=10)
train_label.grid(row=1, column=0, padx=10, pady=10)
#let's put default text(histoire de pas avoir un truc vide)
ayit = ['KNN', "Random forest", "SVM", "LDA", "Logistic regression"]
accuracy = [0, 0, 0, 0, 0]

printage = ["training_results:"]

for i, result_t in enumerate(accuracy):
    stars = int(result_t * 5)  # Scale accuracy to 5 stars
    rating = "★" * stars + "☆" * (5 - stars)  
    percentage = f"({result_t * 100:.2f}%)"
    printage.append(f"{ayit[i]}: {rating} {percentage}")

train_label.config(text='\n'.join(printage))

test_frame = ttk.LabelFrame(classif_frame, text="testing results", padding=(20, 10))
test_frame.grid(row=3, column=0, padx=(20, 10), pady=10, sticky="nsew")
test_label = tk.Label(test_frame, text='', padx=10, pady=10)
test_label.grid(row=1, column=0, padx=10, pady=10)
#put default text 
printage1 = ["testing_results:"]
for i, result_t in enumerate(accuracy):
    stars = int(result_t * 5)  # Scale accuracy to 5 stars
    rating = "★" * stars + "☆" * (5 - stars)  # Display stars and empty stars
    percentage = f"({result_t * 100:.2f}%)"
    printage1.append(f"{ayit[i]}: {rating} {percentage}")

test_label.config(text='\n'.join(printage1))

progress = ttk.Progressbar(classif_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
progress.grid(row=1, column=0, padx=10, pady=10)

# # Create a button to start the classification
# start_button = tk.Button(root, text="Start", command=lambda: threading.Thread(target=classification_thread).start())
# start_button.pack()

classify_button = tk.Button(classif_frame, text="Classify", command=classify_folder)
classify_button.grid(row=0, column=0, padx=10, pady=10)
#------------------------------------------------------------------------------------------------------------------
# frame for mouvement
metrics_frame = ttk.LabelFrame(root, text="Metrics", padding=(20, 10))
metrics_frame.grid(row=1, column=1, padx=(20, 10), pady=10, sticky="nsew",columnspan=2)
#hna we select teh classifier we want 
Classifier_label = ttk.Label(metrics_frame, text="Classifier Used:")
Classifier_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
Classifier_variable = tk.StringVar(value="SELECT Classifier")
Classifier_dropdown = ttk.OptionMenu(metrics_frame, Classifier_variable,"SVM", "SVM","LDA", "RF", "KNN","Logistic regression")
Classifier_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="we")


#plot confusion metric 

for i in range(5):
    matrix_label = tk.Label(metrics_frame, text=f"Confusion Matrix of {i+1}:")
    matrix_label.grid(row=1, column=i+1, padx=5, pady=5, sticky="w")
    matrix_text = tk.Text(metrics_frame, width=50, height=5)
    matrix_text.insert(tk.END, "No Data")
    matrix_text.configure(state='disabled')
    matrix_text.grid(row=2, column=i+1, padx=5, pady=5, sticky="w")



#------------------------------------------------------------------------------------------------------------------
# Center the window, and set minsize
root.update()
root.minsize(root.winfo_width(), root.winfo_height())
x_cordinate = int((root.winfo_screenwidth()/2) - (root.winfo_width()/2))
y_cordinate = int((root.winfo_screenheight()/2) - (root.winfo_height()/2))
root.geometry("+{}+{}".format(x_cordinate, y_cordinate))

# Start the main loop
root.mainloop()

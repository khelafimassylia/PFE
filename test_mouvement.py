import os
import serial
import time
# import csv

# ser = serial.Serial('COM5', 115200)
# movement_names = ["HC", "HO", "T", "I", "M"] #handclose hand open thumb  index middle
# movement_durations = 5 #in seconds
# pause_duration = 3  #in seconds
# repeat_count = 10 #hna ça depend ida nedou plus ou moins
# sample_rate = 1000  #Sample rate in Hz

# for movement_index, movement_name in enumerate(movement_names):#tssema for each mouvement ilopi hadi
#     movement_folder = movement_name #hna create a folder with the name of the mouvement 
#     os.makedirs(movement_folder, exist_ok=True) 
#     for repeat_index in range(repeat_count):
#         file_name = f"{movement_name}_{repeat_index+1}.csv"
#         file_path = os.path.join(movement_folder, file_name) 
#         with open(file_path, mode='w', newline='') as csv_file:
#             fieldnames = ['value']
#             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#             #writer.writeheader()
#             print(f"Beginning movement {movement_name} (repeat {repeat_index + 1})")
#             #time.sleep(1)
#             start_time = time.time()
#             time.sleep(movement_durations)
#             #end_time = time.time()
#             #start_time = time.time()
#             movement_samples_written = 0
#             while movement_samples_written < movement_durations * sample_rate:
#                 data = ser.readline().decode().rstrip()
#                 values = data.split(',')
#                 if data:
#                     writer.writerow({'value': data})
#                     movement_samples_written += 1
                    
#             end_time = time.time()
            
#             print(f"Movement {movement_name} (repeat {repeat_index + 1}) complete. Elapsed time: {end_time - start_time:.2f} s")
#             #time.sleep(pause_duration)
#             if repeat_index <= repeat_count - 1:
#                  print(f"Pause for {pause_duration} seconds")
#                  time.sleep(pause_duration)
# print("All movements complete")
# test 2 electrodes
import os
import serial
import time
import csv
 
ser = serial.Serial('COM5', 115200)
movement_names = ["HC", "HO", "T", "I", "M","R","P"] #handclose hand open thumb  index middle
movement_durations = 5 #in seconds
pause_duration = 3  #in seconds
repeat_count = 10 #hna ça depend ida nedou plus ou moins
sample_rate = 1000  #Sample rate in Hz

for movement_index, movement_name in enumerate(movement_names):#tssema for each mouvement ilopi hadi
     movement_folder = movement_name #hna create a folder with the name of the mouvement 
     os.makedirs(movement_folder, exist_ok=True) 
     for repeat_index in range(repeat_count):
         file_name = f"{movement_name}_{repeat_index+1}.csv"
         file_path = os.path.join(movement_folder, file_name) 
         with open(file_path, mode='w', newline='') as csv_file:
             fieldnames = ['value1', 'value2']
             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
             #writer.writeheader()
             print(f"Beginning movement {movement_name} (repeat {repeat_index + 1})")
             #time.sleep(1)
             start_time = time.time()
             time.sleep(movement_durations)
             ##end_time = time.time()
            ##start_time = time.time()
             movement_samples_written = 0
             while movement_samples_written < movement_durations * sample_rate:
                 data = ser.readline().decode().rstrip()
                 values = data.split(',')
                 if len(values) == 2:  
                   writer.writerow({'value1': values[0], 'value2': values[1]})
                   movement_samples_written += 1
             end_time = time.time()
            
             print(f"Movement {movement_name} (repeat {repeat_index + 1}) complete. Elapsed time: {end_time - start_time:.2f} s")
             #time.sleep(pause_duration)
             if repeat_index <= repeat_count - 1:
                  print(f"Pause for {pause_duration} seconds")
                  time.sleep(pause_duration)
print("All movements complete")

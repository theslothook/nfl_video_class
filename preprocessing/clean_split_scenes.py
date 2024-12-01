import pandas as pd
import os
import shutil
import subprocess
import glob



def clean_split_scenes(game):
    game = 'DET_DAL'
    directory = 'clips/' + game
    temp_directory = os.path.join(directory, 'temp')
    new_directory = 'clips/' + game + '_1'

    full_files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    counter = 1
    error_files = []
    # Loop through each file in the directory
    for index, filename in enumerate(full_files, start=1):
        # files.append(filename)
        file_path = os.path.join(directory, filename)
        # Check if it's a file and if the size is greater than 13 MB (13 * 1024 * 1024 bytes)
        # Check if it's a file and if the size is greater than 13 MB
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 13 * 1000*1000:
            # Copy the file to the temp directory
            if not os.path.exists(temp_directory):
                os.makedirs(temp_directory)
            # if not os.makedirs(directory, exist_ok=True)
            shutil.copy(file_path, temp_directory)
            # # Remove the original file

            process_file = os.path.join(temp_directory, filename)

            print("Hello")

            command = [
                "scenedetect",
                "--input", process_file,
                "-s", "30",
                "split-video",
                "--output", temp_directory + '/',
                "detect-adaptive",
                "--threshold", "2"
            ]

            try:
                subprocess.run(command, check=True)
                print("Command ran successfully.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred: {e}")

            files = glob.glob(os.path.join(temp_directory, "**"), recursive=True)

            if (min([os.path.getsize(file) for file in files if os.path.isfile(file)]) / 1000000) >= 3.0:

                if (len([f for f in os.listdir(temp_directory)]) > 2
                        and len([f for f in os.listdir(temp_directory)]) < 4):

                    ff = sorted([f for f in os.listdir(temp_directory)
                                 if os.path.isfile(os.path.join(temp_directory, f))])
                    for f in ff:
                        temp_file_path = os.path.join(temp_directory, f)
                        if len(f.split("-")) <= 3:
                            os.remove(temp_file_path)


                        else:
                            # shutil.move(temp_file_path, directory)
                            # os.remove(file_path)
                            new_filename = game + '-Scene-' + str(counter).zfill(3) + '.mp4'
                            old_path = os.path.join(directory, f)
                            new_path = os.path.join(new_directory, new_filename)
                            shutil.copy(temp_file_path, new_path)

                            counter += 1
                elif (len([f for f in os.listdir(temp_directory)]) > 2 and
                      len([f for f in os.listdir(temp_directory)]) > 3):
                    error_files.append(filename)
                    counter += 1
                else:
                    new_filename = game + '-Scene-' + str(counter).zfill(3) + '.mp4'
                    new_path = os.path.join(new_directory, new_filename)
                    shutil.copy(file_path, new_path)
                    counter += 1
                shutil.rmtree(temp_directory)
                os.makedirs(temp_directory)
            else:
                new_filename = game + '-Scene-' + str(counter).zfill(3) + '.mp4'
                new_path = os.path.join(new_directory, new_filename)
                shutil.copy(file_path, new_path)
                counter += 1
            shutil.rmtree(temp_directory)
            os.makedirs(temp_directory)
        else:
            new_filename = game + '-Scene-' + str(counter).zfill(3) + '.mp4'
            new_path = os.path.join(new_directory, new_filename)
            shutil.copy(file_path, new_path)
            counter += 1

    print(error_files)
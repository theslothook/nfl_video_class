import pandas as pd
import os
import shutil
import subprocess


def tag_scenes_with_labels(game):
    game = 'BUF_HOU'

    directory = 'clips/' + game
    temp_directory = os.path.join(directory, 'temp')
    new_directory = 'clips/' + game + '_1'

    df = pd.read_csv('../Documents/Bills_Texans_Charted.csv')
    # df = df[1:]
    # df = df[df['Location'].isna() == False]
    # df['has_no_play'] = df['Detail'].str.contains(r"\(no play\)", na=False)

    df.columns.values[5] = 'Team1'
    df.columns.values[6] = 'Team2'
    df.columns.values[11] = 'OFormation'
    df.columns.values[12] = 'DFormation'

    df = df[df['Team1'].isna()==False]

    file_count = len([f for f in os.listdir(new_directory) if os.path.isfile(os.path.join(new_directory, f))])

    df['Screen'] = df['Screen'].fillna(0)
    df["Screen"] = df["Screen"].replace(" ", 0).astype(int)
    df['Pressure'] = df['Pressure'].fillna(0)
    df["Pressure"] = df["Pressure"].replace(" ", 0).astype(int)
    df['Shotgun'] = df['Shotgun'].fillna(0)
    df["Shotgun"] = df["Shotgun"].replace(" ", 0).astype(int)
    df['PlayAction'] = df['PlayAction'].fillna(0)
    df["PlayAction"] = df["PlayAction"].replace(" ", 0).astype(int)
    df['OFormation'] = df['OFormation'].fillna("ST")
    df = df[df['Detail'].str.contains("Unsportsmanlike Conduct", case=False, na=False) == False]
    full_files = sorted([f for f in os.listdir(new_directory) if os.path.isfile(os.path.join(new_directory, f))])

    assert len(df) * 2 == file_count, "there is a file mismatch"

    # Loop through each file in the directory
    for index, filename in enumerate(full_files, start=0):
        file_path = os.path.join(new_directory, filename)


        # ## NEED to fix a scene ####
        # if int(file_path.split("/")[-1].split("-")[-1].split(".")[0]) > 162:
        #     temp_file = ("-".join(file_path.split("-")[:-1]) + "-" +
        #                  str(int(file_path.split("/")[-1].split("-")[-1].split(".")[0]) - 1) + ".mp4")
        #     os.rename(file_path, temp_file)

        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        shotgun = 'NoSG'
        playaction = 'NoPA'
        screen = 'NoSc'
        print(index)
        # if df['OFormation'].iloc[index//2] == 'ST':
        #     file_ext = 'Yes' + "_" + shotgun + "_" + playaction + "_" + screen

        if df['Shotgun'].iloc[index//2] == 1:
            shotgun = 'SG'
        if df['PlayAction'].iloc[index//2] == 1:
            playaction = "PA"
        if df['Screen'].iloc[index//2] == 1:
            screen = "SC"
        file_ext = 'No' + "_" + shotgun + "_" + playaction + "_" + screen
        new_filename = filename.split(".")[0] + "_" + file_ext + '.mp4'
        new_path = os.path.join(new_directory, new_filename)
        shutil.copy(file_path, new_path)
        # os.rename(file_path, new_path)

import os
import shutil


def move_files_to_videos(folder_list, destination="videos"):
    # Ensure the destination folder exists
    if not os.path.exists(destination):
        os.makedirs(destination)

    for folder in folder_list:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        # List all files in the current folder
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)

            # Check if it's a file
            if os.path.isfile(file_path):
                # Move the file to the destination folder
                try:
                    shutil.move(file_path, destination)
                    print(f"Moved: {file_path} -> {destination}")
                except Exception as e:
                    print(f"Error moving file {file_path}: {e}")

    print(f"All files moved to '{destination}'.")


# # Example usage
dir = 'clips/'
folders_to_process = [dir + "BUF_HOU_1", dir + "BAL_WAS_1", dir + "DET_DAL_1", dir + "MN_GB_1"]
move_files_to_videos(folders_to_process, dir + "videos")

import cv2


def count_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video
    video.release()

    return total_frames


# Example usage
video_path = "clips/BAL_WAS/BAL_WAS-Scene-229.mp4"
frame_count = count_frames(video_path)
print(f"Total number of frames in the video: {frame_count}")
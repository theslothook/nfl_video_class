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

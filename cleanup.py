import os
import shutil


def remove_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Removes directories and their contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


directories_to_clean = ['captures/photos', 'captures/videos']

for directory in directories_to_clean:
    if os.path.exists(directory):
        remove_files_in_directory(directory)
        print(f'Cleaned up {directory}.')
    else:
        print(f'Directory {directory} does not exist.')

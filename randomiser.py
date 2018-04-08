import os
import random
import shutil


def split_files_into_groups(dir_path, N):
    """
    Randomly and evenly assigns a file in the dir path to the N groups
    :param dir_path: string with the dir path holding the files
    :param N: integer, number of groups to split in
    :return: None
    """
    files = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, file))]
    available_files = len(files)

    for group in range(1, N+1):
        target_dir = os.path.join(dir_path, "group_{}".format(group))
        if os.path.isdir(target_dir) is False:
            os.makedirs(target_dir)

        number_of_files_in_group = int(len(files) / N)
        number_of_files_in_group += 1 if group <= len(files) % N else 0

        for i in range(0, number_of_files_in_group):
            random_file_index = random.randint(0, available_files-1)  # Select a random available file
            selected_file = files[random_file_index]
            # Now swap selected file to put it at the end
            files[random_file_index], files[available_files-1] = files[available_files-1], files[random_file_index]
            # print("Selected index {}, now swapping indexes {} and {}".format(random_file_index,
            #                                                                 random_file_index, available_files-1))
            available_files -= 1

            # Now copy the file into the target dir
            shutil.copy2(os.path.join(dir_path, selected_file), os.path.join(target_dir, selected_file))
            # print("Copying: {} into {}".format(os.path.join(dir_path, selected_file),
            #                                   os.path.join(target_dir, selected_file)))

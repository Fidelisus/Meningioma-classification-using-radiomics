"""
The script can be used to resturcture the data that I was given to BIDS format.
Here the code quality is lover, as it's a single-use script.
"""

import os
import shutil
import pandas as pd
import errno, os, stat, shutil

if __name__ == "__main__":
    main_folder = "M1"

    # rename directories and move them co BIDS format

    for file in os.listdir(main_folder):
        full_folder_name = os.path.join(main_folder, file)
        os.rename(full_folder_name, os.path.join(main_folder, main_folder + "-sub-" + file))

    all_files = os.listdir(full_folder_name)

    os.mkdir(os.path.join(full_folder_name, "anat"))

    for file in all_files:
        full_file_name = os.path.join(full_folder_name, file)
        print(full_file_name)
        shutil.move(full_file_name, os.path.join(full_folder_name, "anat", file))

    # clean the names and files
    for file in os.listdir(main_folder):
        full_folder_name = os.path.join(main_folder, file)

        all_files = os.listdir(full_folder_name)


        for dir in all_files:
            files_in_dir = 0
            for file in os.listdir(os.path.join(full_folder_name, dir)):
                files_in_dir += 1
                full_file_name = os.path.join(full_folder_name, dir, file)
                # print(full_file_name)
                if full_file_name.endswith("Store"):
                    os.remove(full_file_name)
                if full_file_name.endswith(".nrrd.nrrd"):
                    os.rename(full_file_name, full_file_name.replace(".nrrd.nrrd", ".nrrd"))
                if full_file_name.endswith("nrrd.nrrd"):
                    os.rename(full_file_name, full_file_name.replace("nrrd.nrrd", ".nrrd"))
                if "image" not in full_file_name and "label" not in full_file_name:
                    print(full_file_name)
    print(files_in_dir)

    # create tsv files
    main_folder = "M3"
    label = 3

    list_of_files = []
    list_of_labels = []

    for a in os.listdir(main_folder):
        full_folder_name = os.path.join(main_folder, a)
        all_files = os.listdir(full_folder_name)

        for dir in all_files:
            # print(os.listdir(os.path.join(full_folder_name, dir)))
            for file in os.listdir(os.path.join(full_folder_name, dir)):
                full_file_name = os.path.join(full_folder_name, dir, file)

                if "image" in full_file_name:
                    list_of_files.append(os.path.join(a, dir, file))
                elif "label" in full_file_name:
                    list_of_labels.append(os.path.join(a, dir, file))
                # else:
                #     print(full_file_name)

    df = pd.DataFrame({
        "file_path": sorted(list_of_files),
        "label_path": sorted(list_of_labels),
        "who_grading": [label]*len(list_of_files),
    })
    print(df)
    df.to_csv(main_folder + "/labels.tsv", index = False, sep="\t")

    # change files names

    df = pd.read_csv(main_folder + "/labels.tsv", sep="\t")
    print(df)
    df["file_path"][df["who_grading"] == 1] = "M1-" + df["file_path"][df["who_grading"] == 1].astype(str)
    df["label_path"][df["who_grading"] == 1] = "M1-" + df["label_path"][df["who_grading"] == 1].astype(str)
    df["file_path"][df["who_grading"] == 2] = "M2-" + df["file_path"][df["who_grading"] == 2].astype(str)
    df["label_path"][df["who_grading"] == 2] = "M2-" + df["label_path"][df["who_grading"] == 2].astype(str)
    df["file_path"][df["who_grading"] == 3] = "M3-" + df["file_path"][df["who_grading"] == 3].astype(str)
    df["label_path"][df["who_grading"] == 3] = "M3-" + df["label_path"][df["who_grading"] == 3].astype(str)
    print(df)
    df.to_csv(main_folder + "/labels.tsv", index = False, sep="\t")

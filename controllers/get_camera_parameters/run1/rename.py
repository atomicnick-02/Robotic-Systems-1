import os

def rename_files_in_folder(folder_path):
	for filename in os.listdir(folder_path):
		file_path = os.path.join(folder_path, filename)
		if os.path.isfile(file_path):
			new_name = f"run1_{filename}"
			new_path = os.path.join(folder_path, new_name)
			os.rename(file_path, new_path)
			print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
	folder = input("Enter the folder path: ").strip()
	rename_files_in_folder(folder)
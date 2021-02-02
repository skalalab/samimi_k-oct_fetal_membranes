import cv2
import os
import matplotlib.pyplot as plt
# print(f'{os.getcwd()}')
working_directory = os.getcwd() + os.sep + 'projects' + os.sep + 'placenta oct (Kayvan)'
os.chdir(working_directory)
# print(f'{os.getcwd()}')
vidcap = cv2.VideoCapture('combined_stacks_mp4.mp4')
success, image = vidcap.read()
count = 0


img_oct = cv2.cvtColor(image[:265, ...], cv2.COLOR_BGR2RGB)
img_labels = cv2.cvtColor(image[265:, ...], cv2.COLOR_BGR2RGB)
plt.imshow(img_oct)  # original
plt.imshow(img_labels)  # masked
plt.show()

while success:

    # save current frame
    cv2.imwrite(f"dataset/images/image_{count}.jpg", img_oct)     # save frame as JPEG file
    cv2.imwrite(f"dataset/labels/labels_{count}.jpg", img_labels)     # save frame as JPEG file

    # read next frame
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    if success:
        img_oct = cv2.cvtColor(image[:265, ...], cv2.COLOR_BGR2RGB)
        img_labels = cv2.cvtColor(image[265:, ...], cv2.COLOR_BGR2RGB)
        count += 1

#%% Process images


directory = "C:/Users/Nabiki/Desktop/development/skala_lab/projects/placenta oct (Kayvan)/dataset/"

dir_images = directory + 'images'
dir_labels = directory + 'labels'


filename_all_images = os.listdir(dir_images)
filename_all_labels = os.listdir(dir_labels)

curr_image = 0
image = cv2.imread(dir_images + os.sep + filename_all_images[curr_image])
labels = cv2.imread(dir_labels + os.sep + filename_all_labels[curr_image])

plt.imshow(image)
plt.imshow(labels)
plt.show()

# Why would this be useful?
# Send the link to github repository and why it would be useful to use


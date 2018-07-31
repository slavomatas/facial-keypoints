import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from models import Net

import torch
from torchvision import transforms, utils


# transforms

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # if image has no grayscale color channel, add one
        if (len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)


# load in color image for face detection
image = cv2.imread('images/obamas.jpg')

# switch red and blue color channels
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plot the image
# fig = plt.figure(figsize=(9,9))
# plt.imshow(image)

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

# make a copy of the original image to plot detections on
image_with_detections = image.copy()

# loop over the detected faces, mark the image where each face is found
for (x, y, w, h) in faces:
    # draw a rectangle around each detected face
    # you may also need to change the width of the rectangle drawn depending on image resolution
    cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)


#fig = plt.figure(figsize=(9, 9))
#plt.imshow(image_with_detections)

## TODO: load the best saved model parameters (by your path name)
net = Net()
net.load_state_dict(torch.load('saved_models/keypoints_model_5_epochs.pt'))

## print out your net and prepare it for testing (uncomment the line below)
net.eval()
print(net)

image_copy = np.copy(image)

# loop over the detected faces from your haar cascade
for (x, y, w, h) in faces:

    # pad = 55
    # roi = image_copy[(y-pad):(y+h+pad), (x-pad):(x+w+pad)]
    # roi = cv2.copyMakeBorder(roi,50,50,50,50,cv2.BORDER_REPLICATE)

    # Select the region of interest that is the face in the image
    roi = image_copy[y:y + h, x:x + w]

    ## TODO: Convert the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = roi / 255.0

    ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    rescale = Rescale(224)
    roi = rescale(roi)

    ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    to_tensor = ToTensor()
    torch_tensor = to_tensor(roi)

    ## TODO: Make facial keypoint predictions using your loaded, trained network
    ## perform a forward pass to get the predicted facial keypoints

    # forward pass to get net output
    torch_tensor = torch_tensor.type(torch.FloatTensor)
    torch_tensor = torch_tensor.unsqueeze(0)
    output_pts = net(torch_tensor)

    # reshape to batch_size x 68 x 2 pts
    output_pts = output_pts.view(output_pts.size()[0], 68, -1)

    # un-transform the predicted key_pts data
    predicted_key_pts = output_pts.data.numpy()
    predicted_key_pts = predicted_key_pts.reshape(-1, 2)

    # undo normalization of keypoints
    predicted_key_pts = predicted_key_pts*50.0+100
    print(predicted_key_pts[:, 0], predicted_key_pts[:, 1])

    ## TODO: Display each detected face and the corresponding keypoints
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(roi)
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')

    # ax = plt.subplot(1, 3, i + 1)
    # plt.tight_layout()
    # ax.set_title(type(tx).__name__)
    # show_keypoints(transformed_sample['image'], transformed_sample['keypoints'])

from model import OneGaussianVideo


if __name__ == '__main__':
    u=OneGaussianVideo() # Reads video, initializes
    u.modeltrainGaussian() # Compute gaussian parameters
    u.classifyTest(alpha=0.5,rho=0.5,isAdaptive=False) # Apply gaussian to frames
    #cv2.imshow('',of.astype(np.uint8)*255)
    #cv2.waitKey()
    #u.state_of_art()

    print('u')

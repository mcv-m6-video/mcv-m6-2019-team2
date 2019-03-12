import cv2
from model import OneGaussianVideo


if __name__ == '__main__':
    u=OneGaussianVideo() # Reads video, initializes
    m=u.modeltrainGaussian() # Compute gaussian parameters

    frame_our = u.classifyTest(alpha=2,rho=0.2,isAdaptive=True) # Apply gaussian to frames
    #cv2.imshow('',of.astype(np.uint8)*255)
    #cv2.waitKey()
    #frame_state =u.state_of_art()
    cv2.destroyAllWindows()
    print('u')
